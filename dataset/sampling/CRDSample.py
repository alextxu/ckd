import numpy as np
from torch.utils.data import Dataset

# from https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/dataset/cifar100.py#L109C49-L109C49

class CRDSample(Dataset):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, original_dataset, sampling_params):
        self.original_dataset = original_dataset
        # defaults: k=4096, mode='exact', is_sample=True, percent=1.0
        k, mode, is_sample, percent, num_classes = sampling_params
        
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        # changed for compatibility
        # num_samples = len(self.original_dataset.dataset.dataset.data)
        # label = self.original_dataset.dataset.dataset.targets
        num_samples = len(self.original_dataset)
        label = [lbl for _, lbl, _ in self.original_dataset]

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        # self.cls_positive = np.asarray(self.cls_positive)
        # self.cls_negative = np.asarray(self.cls_negative)
    
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        img, target, logit_t = self.original_dataset[index]

        if not self.is_sample:
            # directly return
            return img, target, logit_t, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, logit_t, index, sample_idx

