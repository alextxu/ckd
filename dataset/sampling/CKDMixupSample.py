import math
import random
import itertools
import torch
from torch.utils.data import Dataset

class CKDMixupSample(Dataset):
    """vs mixup ablation"""

    def __init__(self, original_dataset, sampling_params):
        self.original_dataset = original_dataset
        pos, neg, max_ckd_samples, n_cls = sampling_params
        
        self.pos = pos
        self.neg = neg
        self.n_cls = n_cls
        num_train_images = len(original_dataset)
        trainset_indices = list(range(num_train_images))
        tot_possible_ckd = math.comb(num_train_images, pos) * math.comb(num_train_images - pos, neg)
        if tot_possible_ckd >= max_ckd_samples * 5:
            all_chosen_indices = set()
            while len(all_chosen_indices) < max_ckd_samples:
                temp = random.sample(trainset_indices, pos + neg)
                temp = sorted(temp[:pos]) + sorted(temp[pos:])
                chosen_indices = tuple(temp)
                if chosen_indices not in all_chosen_indices:
                    all_chosen_indices.add(chosen_indices)
            all_chosen_indices = list(all_chosen_indices)
        else:
            all_combos = list(itertools.combinations(trainset_indices, pos + neg))
            all_pos_combos = list(itertools.combinations(list(range(pos + neg)), pos))
            all_ckd_perms = []
            for combo in all_combos:
                for pos_combo in all_pos_combos:
                    pos_indices = []
                    neg_indices = []
                    for i in range(pos + neg):
                        if i in pos_combo:
                            pos_indices.append(combo[i])
                        else:
                            neg_indices.append(combo[i])
                    sample = pos_indices + neg_indices
                    all_ckd_perms.append(sample)
            if len(all_ckd_perms) <= max_ckd_samples:
                all_chosen_indices = all_ckd_perms
            else:
                perm_indices = random.sample(list(range(len(all_ckd_perms))), max_ckd_samples)
                all_chosen_indices = []
                for idx in perm_indices:
                    all_chosen_indices.append(all_ckd_perms[idx])
        self.indices = all_chosen_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_indices = self.indices[idx]
        samples = []
        teacher_logits = []
        labels = []
        for i in sample_indices:
            img, lbl, logit_t = self.original_dataset[i]
            samples.append(img.unsqueeze(0))
            teacher_logits.append(logit_t.unsqueeze(0))
            labels.append(lbl)
        pos = torch.cat(samples[:self.pos])
        neg = torch.cat(samples[self.pos:])
        label_pos = torch.zeros(self.n_cls)
        label_neg = torch.zeros(self.n_cls)
        teacher_logits_pos = torch.mean(torch.cat(teacher_logits[:self.pos]), axis=0)
        teacher_logits_neg = torch.mean(torch.cat(teacher_logits[self.pos:]), axis=0)
        for i in range(self.pos):
            label_pos[labels[i]] += 1 / self.pos
        for i in range(self.pos, len(labels)):
            label_neg[labels[i]] += 1 / self.neg
        return pos, neg, label_pos, label_neg, teacher_logits_pos, teacher_logits_neg