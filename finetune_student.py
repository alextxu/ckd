from __future__ import print_function

import os
import argparse
import time
import wandb

# import tensorboard_logger as tb_logger
import torch
import torch.cuda as cuda
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders_ft

from helper.util import early_stopping_adjust_lr, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate

import random
import numpy as np

def get_config_str(configs):
    important_configs = ['epochs', 'patience', 'learning_rate', 'lr_decay_rate', 'dataset', 'model', 'subset_size', 'trial', 'seed']
    important_info = [str(configs[c]) for c in important_configs]
    return '_'.join(important_info)

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # wandb
    parser.add_argument('--_tags', type=str, default='debug', help='wandb tags')
    parser.add_argument('--wdb_project', type=str, default='xxx', help='wandb project name')
    parser.add_argument('--wdb_entity', type=str, default='xxx', help='wandb entity name')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--lr_decay_times', type=int, default=3, help='number of times to decay lr')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', ])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    parser.add_argument('--subset_size', type=int, default=50000, help='subset size')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = './save/models'
    opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}_subset_{}'.format(opt.model, opt.dataset, opt.learning_rate, opt.weight_decay, opt.trial, opt.subset_size)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    return opt

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    best_acc = -1
    test_acc = 0
    test_acc_top5 = 0
    test_loss = 1e9
    last_improve = 0
    times_lr_adjusted = 0

    opt = parse_option()

    set_random_seed(opt.seed)

    # create run on wandb
    run = wandb.init(
        project=opt.wdb_project,
        entity=opt.wdb_entity, 
        config=vars(opt),
        tags=[t.strip() for t in opt._tags.split(',')],
    )

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader, test_loader = get_cifar100_dataloaders_ft(batch_size=opt.batch_size, num_workers=opt.num_workers, subset_size=opt.subset_size)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = model_dict[opt.model](num_classes=n_cls)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    for epoch in range(1, opt.epochs + 1):
        if times_lr_adjusted > opt.lr_decay_times:
            break

        early_stopping_adjust_lr(times_lr_adjusted, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        run.log({'train_acc': train_acc, 'train_loss': train_loss}, step=epoch)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # logger.log_value('train_acc', train_acc, epoch)
        # logger.log_value('train_loss', train_loss, epoch)

        val_acc, val_acc_top5, val_loss = validate(val_loader, model, criterion, opt)
        run.log({'val_acc': val_acc, 'val_acc_top5': val_acc_top5, 'val_loss': val_loss}, step=epoch)

        # logger.log_value('test_acc', test_acc, epoch)
        # logger.log_value('test_acc_top5', test_acc_top5, epoch)
        # logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if val_acc > best_acc:
            last_improve = epoch
            best_acc = val_acc
            test_acc, test_acc_top5, test_loss = validate(test_loader, model, criterion, opt)
            temp_path = f"temp/finetune_{get_config_str(run.config)}.pth"
            torch.save(model.state_dict(), temp_path)
            print('saving the best model!')
        
        if epoch - last_improve >= opt.patience:
            last_improve = epoch
            temp_path = f"temp/finetune_{get_config_str(run.config)}.pth"
            model.load_state_dict(torch.load(temp_path))
            times_lr_adjusted += 1

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)
    print('test accuracy:', test_acc)

    run.log({'test_acc': test_acc, 'test_acc_top5': test_acc_top5, 'test_loss': test_loss, 'best_val_acc': best_acc})


if __name__ == '__main__':
    main()
