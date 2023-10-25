import random
import torch
from torch.utils.data import Subset, Dataset

class CachedSubset(Dataset):
    """Takes a subset of a given dataset and caches the teacher output logits to each sample"""
    def __init__(self, dataset, indices, teacher, preact=False):
        self.dataset = dataset
        self.indices = indices
        self.cached_logits = {}
        with torch.no_grad():
            for idx in indices:
                img, _ = self.dataset[idx]
                img = img.unsqueeze(0).cuda()
                logit_t = teacher(img, is_feat=False, preact=preact)
                self.cached_logits[idx] = logit_t.squeeze(0).detach().cpu()
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] + (self.cached_logits[self.indices[i]],) for i in idx]]
        return self.dataset[self.indices[idx]] + (self.cached_logits[self.indices[idx]],)

class CachedSubsetWithInput(Dataset):
    """Takes a subset of a given dataset and caches the teacher output logits to each sample"""
    def __init__(self, dataset, indices, teacher, preact=False):
        self.dataset = dataset
        self.indices = indices
        self.cached_logits = {}
        self.cached_inputs = {}
        with torch.no_grad():
            for idx in indices:
                img, _ = self.dataset[idx]
                self.cached_inputs[idx] = img.detach()
                img = img.unsqueeze(0).cuda()
                logit_t = teacher(img, is_feat=False, preact=preact)
                self.cached_logits[idx] = logit_t.squeeze(0).detach().cpu()
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] + (self.cached_logits[self.indices[i]], self.cached_inputs[self.indices[i]]) for i in idx]]
        return self.dataset[self.indices[idx]] + (self.cached_logits[self.indices[idx]], self.cached_inputs[self.indices[idx]])

def train_val_split(train_set, val_set, train_percent=0.8, fixed_val_size=0):
    if fixed_val_size != 0:
        train_len = len(train_set) - fixed_val_size
    else:
        train_len = int(len(train_set) * train_percent)
    shuffled_indices = list(range(len(train_set)))
    random.shuffle(shuffled_indices)
    train_split = Subset(train_set, shuffled_indices[:train_len])
    val_split = Subset(val_set, shuffled_indices[train_len:])
    return train_split, val_split

def mix_func(relational, lam, x, y):
    if relational == 'ckd':
        return x - y
    elif relational == 'ckd_add':
        return 0.5 * x + 0.5 * y
    elif relational == 'ckd_mixup':
        return lam * x + (1 - lam) * y
    else:
        raise NotImplementedError('unknown mix func')