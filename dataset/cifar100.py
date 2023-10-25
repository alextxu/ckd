from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image
import copy
import random
import torch
from .util import train_val_split, CachedSubset, CachedSubsetWithInput
from dataset.sampling.CKDSample import CKDSample
from dataset.sampling.MixupSample import MixupSample
from dataset.sampling.MixupThreeSample import MixupThreeSample
from dataset.sampling.CRDSample import CRDSample
from dataset.sampling.RKDSampler import RKDSampler
from dataset.sampling.CKDMixupSample import CKDMixupSample
from dataset.sampling.InterCKDSample import InterCKDSample

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""

MEAN = (0.5071, 0.4867, 0.4408)
STD  = (0.2675, 0.2565, 0.2761)

def get_data_folder():
    """
    return server-dependent path to store the data
    """
    data_folder = '../data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_cifar100_dataloaders(model_t, batch_size=128, num_workers=8, is_instance=False,
                             subset_size=50000, relational='', relational_params=None,
                             preact=False, use_DA=2, distill='kd'):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    # from cutmixpick
    transforms_ = []
    if use_DA > 0:
        transforms_ += [transforms.RandomCrop(32, padding=4)]
    if use_DA > 1:
        transforms_ += [transforms.RandomHorizontalFlip()]
    transforms_ += [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
    ]
    train_transform = transforms.Compose(transforms_)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    if is_instance:
        train_set = CIFAR100Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=train_transform)
    else:
        train_set = datasets.CIFAR100(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    
    # take subset and form validation set
    if subset_size == 0 or subset_size > len(train_set):
        return NotImplementedError("Invalid subset size")
    
    subset_indices = random.sample(list(range(len(train_set))), subset_size)

    if distill != 'kd':
        train_subset = CachedSubsetWithInput(train_set, subset_indices, model_t, preact)
    else:
        train_subset = CachedSubset(train_set, subset_indices, model_t, preact)
    val_set = copy.copy(train_set)
    val_set.transform = test_transform
    val_subset = Subset(val_set, subset_indices)
    train_split, val_split = train_val_split(train_subset, val_subset)
    n_data = len(train_split)

    # special sampling methods for relational techniques
    batch_sampler = None
    if relational == 'ckd' or relational == 'ckd_mixup' or relational == 'ckd_add':
        relational_train = CKDSample(train_split, relational_params)
    elif relational == 'ckd_inter':
        relational_train = InterCKDSample(train_split, relational_params)
    elif relational == 'ckd_mixup':
        relational_train = CKDMixupSample(train_split, relational_params)
    elif relational == 'mixup' or relational == 'cutmix':
        relational_train = MixupSample(train_split, relational_params)
    elif relational == 'mixup3':
        relational_train = MixupThreeSample(train_split, relational_params)
    elif relational == 'crd':
        relational_train = CRDSample(train_split, relational_params)
    elif relational == 'rkd':
        iter_per_epoch = relational_params // batch_size
        batch_sampler = RKDSampler(train_split, batch_size, m=5, iter_per_epoch=iter_per_epoch)
        relational_train = train_split
    else:
        relational_train = train_split

    if batch_sampler != None:
        train_loader = DataLoader(relational_train,
                                  batch_sampler=batch_sampler,
                                  num_workers=num_workers)
    else:
        train_loader = DataLoader(relational_train,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
    val_loader = DataLoader(val_split,
                            batch_size=int(batch_size/2),
                            shuffle=False,
                            num_workers=int(num_workers/2))

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, val_loader, test_loader, n_data


def get_cifar100_dataloaders_ft(batch_size=128, num_workers=8, subset_size=50000):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    # from cutmixpick
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_set = datasets.CIFAR100(root=data_folder,
                                  download=True,
                                  train=True,
                                  transform=train_transform)
    
    # take subset and form validation set
    if subset_size == 0 or subset_size > len(train_set):
        return NotImplementedError("Invalid subset size")
    
    subset_indices = random.sample(list(range(len(train_set))), subset_size)
    train_subset = Subset(train_set, subset_indices)
    val_set = copy.copy(train_set)
    val_set.transform = test_transform
    val_subset = Subset(val_set, subset_indices)
    train_split, val_split = train_val_split(train_subset, val_subset)
    n_data = len(train_split)

    train_loader = DataLoader(train_split,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_split,
                            batch_size=int(batch_size/2),
                            shuffle=False,
                            num_workers=int(num_workers/2))

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, val_loader, test_loader

def get_cifar100_dataloaders_teacher(batch_size=128, num_workers=8):
    data_folder = '../data/'
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_set = datasets.CIFAR100(root=data_folder,
                                  download=True,
                                  train=True,
                                  transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)

    return train_loader, test_loader
