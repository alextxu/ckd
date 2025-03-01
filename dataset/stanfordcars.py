"""Transformation values from https://github.com/lenscloth/RKD/blob/0a6c3c0c190722d428322bf71703c0ae86c25242/run_distill.py"""

from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.utils.data as data
import torch
from torchvision import datasets, transforms
from PIL import Image
import random
import copy
from .util import CachedSubset, train_val_split
from .sampling.CRDSample import CRDSample
from .sampling.CKDSample import CKDSample
from .sampling.MixupSample import MixupSample
from .sampling.RKDSampler import RKDSampler

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_stanfordcars_dataloaders(model_t, batch_size=64, num_workers=8, is_instance=False, subset_size=8144,
                                 relational='', relational_params=None, preact=False, use_DA=2):
    data_folder = '../data/'

    # from cutmixpick
    transforms_ = [transforms.Resize((64, 64))]
    if use_DA > 0:
        transforms_ += [transforms.RandomCrop(56)]
    if use_DA > 1:
        transforms_ += [transforms.RandomHorizontalFlip()]
    transforms_ += [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
    ]
    train_transform = transforms.Compose(transforms_)
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop(56),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # train set and loader
    train_set = datasets.StanfordCars(root=data_folder,
                                      split='train',
                                      transform=train_transform,
                                      download=True)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    
    # take subset and form validation set
    if subset_size == 0 or subset_size > len(train_set):
        return NotImplementedError("Invalid subset size")
    
    subset_indices = random.sample(list(range(len(train_set))), subset_size)

    train_subset = CachedSubset(train_set, subset_indices, model_t, preact)
    val_set = copy.copy(train_set)
    val_set.transform = test_transform
    val_subset = Subset(val_set, subset_indices)
    train_split, val_split = train_val_split(train_subset, val_subset)
    n_data = len(train_split)

    # special sampling methods for relational techniques
    batch_sampler = None
    if relational == 'ckd':
        relational_train = CKDSample(train_split, relational_params)
    elif relational == 'mixup':
        relational_train = MixupSample(train_split, relational_params)
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

    # test set and loder
    test_set = datasets.StanfordCars(root=data_folder,
                                     split='test',
                                     transform=test_transform,
                                     download=True)
    test_loader = DataLoader(test_set,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader, n_data


def get_stanfordcars_dataloaders_teacher(batch_size=128, num_workers=8, subset_size=8144):
    data_folder = '../data/'
    
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomCrop(56),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop(56),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    train_set = datasets.StanfordCars(root=data_folder,
                                      split='train',
                                      transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.StanfordCars(root=data_folder,
                                     split='test',
                                     transform=test_transform)
    test_loader = DataLoader(test_set,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)

    return train_loader, test_loader
