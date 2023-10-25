import math
import random
import itertools
from torch.utils.data import Dataset
import torch

class MixupSample(Dataset):
    """Mixup dataset."""

    def __init__(self, original_dataset, sampling_params):
        self.original_dataset = original_dataset
        max_ckd_samples, n_cls = sampling_params

        self.n_cls = n_cls
        num_train_images = len(original_dataset)
        trainset_indices = list(range(num_train_images))
        tot_possible = math.perm(num_train_images, 2)
        if tot_possible >= max_ckd_samples * 5:
            all_chosen_indices = set()
            while len(all_chosen_indices) < max_ckd_samples:
                temp = random.sample(trainset_indices, 2)
                chosen_indices = tuple(temp)
                if chosen_indices not in all_chosen_indices:
                    all_chosen_indices.add(chosen_indices)
            all_chosen_indices = list(all_chosen_indices)
        else:
            all_combos = list(itertools.combinations(trainset_indices, 2))
            all_pos_combos = list(itertools.combinations([0, 1], 1))
            all_perms = []
            for combo in all_combos:
                for pos_combo in all_pos_combos:
                    pos_indices = []
                    neg_indices = []
                    for i in range(2):
                        if i in pos_combo:
                            pos_indices.append(combo[i])
                        else:
                            neg_indices.append(combo[i])
                    sample = pos_indices + neg_indices
                    all_perms.append(sample)
            if len(all_perms) <= max_ckd_samples:
                all_chosen_indices = all_perms
            else:
                perm_indices = random.sample(list(range(len(all_perms))), max_ckd_samples)
                all_chosen_indices = []
                for idx in perm_indices:
                    all_chosen_indices.append(all_perms[idx])
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
            samples.append(img)
            teacher_logits.append(logit_t)
            labels.append(lbl)
        return samples[0], samples[1], labels[0], labels[1], teacher_logits[0], teacher_logits[1]
