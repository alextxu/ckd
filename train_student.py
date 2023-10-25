"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import time
import wandb
import random
import numpy as np

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders #, get_cifar100_dataloaders_sample

from helper.util import adjust_learning_rate, early_stopping_adjust_lr, get_config_str

from distiller_zoo import DistillKL, HintLoss, Correlation, VIDLoss, RKDLoss, DIST
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate

from helper.TrainHelpers import TrainHelpers
from helper.CRDTrainHelpers import CRDTrainHelpers
from helper.CKDTrainHelpers import CKDTrainHelpers
from helper.MixupTrainHelpers import MixupTrainHelpers
from helper.MixupThreeTrainHelpers import MixupThreeTrainHelpers
from helper.InterTrainHelpers import InterTrainHelpers
from helper.InterCKDTrainHelpers import InterCKDTrainHelpers
from helper.CutMixHelpers import CutMixHelpers

w_rel_defaults = {
    'crd': 0.8,
    'rkd': 1,
    'ckd': 1,
    'ckd_add': 1,
    'ckd_mixup': 1,
    'ckd_inter': 1,
    'dist': 2,
    'mixup': 1,
    'mixup3': 1,
    'cutmix': 1
}

w_inter_defaults = {
    'kd': 0,
    'hint': 100,
    'correlation': 0.02,
    'vid': 1,
}


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # wandb
    parser.add_argument('--_tags', type=str, default='debug', help='wandb tags')
    parser.add_argument('--wdb_project', type=str, default='xxx', help='wandb project name')
    parser.add_argument('--wdb_entity', type=str, default='xxx', help='wandb entity name')

    # basic
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--lr_decay_times', type=int, default=3, help='number of times to decay lr')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    parser.add_argument('--subset_size', type=int, default=50000, help='subset size')
    parser.add_argument('--pretrained_subset_size', type=int, default=0, help='subset size')
    parser.add_argument('--train_val_frac', type=float, default=0.8, help='fraction of training samples for train/val split')
    parser.add_argument('--use_DA', type=int, default=2, help='use flip and crop (0 none, 1 only crop, 2 flip and crop)')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44',
                                 'resnet56', 'resnet110', 'resnet8x4', 'resnet32x4',
                                 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--model_t', type=str, default=None)
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--relational', type=str, default=None,
                        choices=[None, 'ckd', 'crd', 'rkd', 'pkt', 'mixup', 'dist', 'mixup3', 'ckd_mixup', 'ckd_add', 'ckd_inter',  'cutmix'])
    parser.add_argument('--distill', type=str, default='kd',
                        choices=['kd', 'hint', 'correlation', 'vid'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('--w_cls', type=float, default=1.0, help='weight for classification')
    parser.add_argument('--w_kd', type=float, default=0.0, help='weight balance for KD')
    parser.add_argument('--w_rel_scale', type=float, default=1.0, help='weight balance scaling for relational losses, to be multiplied by default')
    parser.add_argument('--w_inter_scale', type=float, default=1.0, help='weight balance for other losses, to be multiplied by default')
    parser.add_argument('--loss_fn', type=int, default=0, help='0 for CE in mixup, 1 for KL in mixup')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    # ckd
    parser.add_argument('--pos_and_neg', default=3, type=int, help='pos + neg samples in ckd')

    # relational sampling
    parser.add_argument('--max_rel_samples', default=100000, type=int, help='max number of relational samples')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01
    
    # scale weight values by defaults
    if opt.relational in w_rel_defaults:
        opt.w_rel = w_rel_defaults[opt.relational] * opt.w_rel_scale
    else:
        opt.w_rel = 0
    opt.w_inter = w_inter_defaults[opt.distill] * opt.w_inter_scale

    # set the path according to the environment
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # get corresponding teacher model if not specified
    if opt.model_t == None:
        opt.model_t = get_corresponding_teacher(opt.model_s)
    
    # fill in the missing path or model name information
    if opt.path_t == None:
        opt.path_t = get_teacher_path(opt.model_t, opt.dataset)
    else:
        opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_R:{}_{}_{}_c:{}_k:{}_r:{}_u:{}_l:{}_s:{}_{}'.format(opt.model_s, opt.model_t, opt.relational, opt.dataset, opt.distill, opt.w_cls, opt.w_kd, opt.w_rel, opt.use_DA, opt.learning_rate, opt.seed, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_path(teacher_name, dataset):
    if dataset == 'cifar100':
        return f'./save/models/{teacher_name}_vanilla/ckpt_epoch_240.pth'
    else:
        raise NotImplementedError('Dataset/teacher combination not supported')


def get_corresponding_teacher(student):
    if student == 'vgg8':
        return 'vgg13'
    elif student == 'resnet32':
        return 'resnet110'
    elif student == 'wrn_16_2':
        return 'wrn_40_2'
    else:
        raise Exception('No corresponding teacher model found')


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    opt = parse_option()

    set_random_seed(opt.seed)

    # create run on wandb
    run = wandb.init(
        project=opt.wdb_project,
        entity=opt.wdb_entity, 
        config=vars(opt),
        tags=[t.strip() for t in opt._tags.split(',')],
    )

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # load teacher model first to cache teacher outputs
    if opt.dataset == 'cifar100':
        n_cls = 100
    else:
        n_cls = 0
    model_t = load_teacher(opt.path_t, n_cls)
    model_t.eval()
    model_t.cuda()

    # dataloader
    if opt.relational == 'ckd' or opt.relational == 'ckd_mixup' or opt.relational == 'ckd_add' or opt.relational == 'ckd_inter':
        neg = opt.pos_and_neg // 2
        pos = opt.pos_and_neg - neg
        relational_params = (pos, neg, opt.max_rel_samples, n_cls)
    elif opt.relational == 'crd':
        relational_params = (opt.nce_k, opt.mode, True, 1.0, n_cls)
    elif opt.relational == 'mixup' or opt.relational == 'mixup3' or opt.relational == 'cutmix':
        relational_params = (opt.max_rel_samples, n_cls)
    elif opt.relational == 'rkd':
        relational_params = opt.max_rel_samples
    else:
        relational_params = None
    preact = False
    if opt.dataset == 'cifar100':
        train_loader, val_loader, test_loader, n_data = \
            get_cifar100_dataloaders(model_t=model_t,
                                     batch_size=opt.batch_size,
                                     num_workers=opt.num_workers,
                                     is_instance=False,
                                     subset_size=opt.subset_size,
                                     relational=opt.relational,
                                     relational_params=relational_params,
                                     preact=preact,
                                     use_DA=opt.use_DA,
                                     distill=opt.distill)
    else:
        raise NotImplementedError(opt.dataset)

    # init student model and other modules
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    if opt.dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32).cuda()
    else:
        raise NotImplementedError(opt.dataset)
    model_s.eval()
    model_s.cuda()
    feat_t, logit_t = model_t(data, is_feat=True)
    feat_s, logit_s = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_val = nn.CrossEntropyLoss().cuda()
    criterion_kd = DistillKL(opt.kd_T).cuda()

    if opt.relational == 'ckd' or opt.relational == 'ckd_dist' or opt.relational == 'ckd_mixup' or opt.relational == 'ckd_add' or opt.relational == 'ckd_inter':
        criterion_cls = DistillKL(opt.kd_T).cuda()
    else:
        criterion_cls = nn.CrossEntropyLoss().cuda()

    if opt.relational == 'mixup' or opt.relational == 'mixup3' or opt.relational == 'cutmix':
        if opt.loss_fn == 0:
            criterion_rel = nn.CrossEntropyLoss().cuda()
        else:
            criterion_rel = DistillKL(opt.kd_T).cuda()
    elif opt.relational == 'ckd' or opt.relational == 'ckd_mixup' or opt.relational == 'ckd_add' or opt.relational == 'ckd_inter':
        criterion_rel = DistillKL(opt.kd_T).cuda()
    elif opt.relational == 'crd':
        opt.s_dim = logit_s.shape[1]
        opt.t_dim = logit_t.shape[1]
        opt.n_data = n_data
        criterion_rel = CRDLoss(opt).cuda()
        module_list.append(criterion_rel.embed_s)
        module_list.append(criterion_rel.embed_t)
        trainable_list.append(criterion_rel.embed_s)
        trainable_list.append(criterion_rel.embed_t)
    elif opt.relational == 'rkd':
        criterion_rel = RKDLoss().cuda()
    elif opt.relational == 'dist':
        criterion_rel = DIST().cuda()
    elif opt.relational == None:
        criterion_rel = None
    else:
        raise NotImplementedError(opt.relational)

    if opt.distill == 'kd':
        criterion_inter = DistillKL(opt.kd_T).cuda()
    elif opt.distill == 'hint':
        criterion_inter = HintLoss().cuda()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'correlation':
        # TODO: check if this is relational
        criterion_inter = Correlation().cuda()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_inter = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        ).cuda()
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_inter)
    else:
        raise NotImplementedError(opt.distill)

    # create a training loop helper object
    helper_classes = {
        'ckd_mixup': CKDTrainHelpers,
        'ckd_add': CKDTrainHelpers,
        'ckd': CKDTrainHelpers,
        'crd': CRDTrainHelpers,
        'mixup': MixupTrainHelpers,
        'mixup3': MixupThreeTrainHelpers,
        'cutmix': CutMixHelpers
    }

    if opt.distill != 'kd':
        if opt.relational == 'ckd_inter':
            helper_class = InterCKDTrainHelpers
        else:
            helper_class = InterTrainHelpers
    else:
        if opt.relational in helper_classes:
            helper_class = helper_classes[opt.relational]
        else:
            helper_class = TrainHelpers
    helper_object = helper_class(opt, criterion_cls, opt.w_cls,
                                 criterion_kd, opt.w_kd,
                                 criterion_rel, opt.w_rel)
    
    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(test_loader, model_t, criterion_val, opt)
    print(f'teacher accuracy: {round(float(teacher_acc), 2)}%')

    # initialize stats before training
    best_acc = -1
    test_acc = 0
    test_acc_top5 = 0
    test_loss = 1e9
    tot_steps = 0
    last_improve = 0
    times_lr_adjusted = 0
    eval_every = (int(opt.subset_size * opt.train_val_frac) + opt.batch_size - 1) // opt.batch_size

    # train until convergence
    while tot_steps // eval_every < opt.epochs and times_lr_adjusted < opt.lr_decay_times:
        torch.cuda.empty_cache()
        early_stopping_adjust_lr(times_lr_adjusted, opt, optimizer)
    # for epoch in range(1, opt.epochs + 1):

        # adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        stats = tot_steps, best_acc, last_improve, test_acc, test_acc_top5, test_loss
        train_loss, stats, done = \
            train(run, helper_object, train_loader, val_loader, test_loader,
                  module_list, trainable_list, criterion_inter, criterion_val,
                  optimizer, opt, stats)
        tot_steps, best_acc, last_improve, test_acc, test_acc_top5, test_loss = stats
        # train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(tot_steps // eval_every, time2 - time1))

        # logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, tot_steps // eval_every)
        run.log({'train_loss': train_loss}, step=tot_steps // eval_every)

        if done:
            # load most recent trainable modules
            last_improve = tot_steps
            for i in range(len(trainable_list)):
                temp_path = f"temp/{get_config_str(run.config)}_{str(i)}.pth"
                trainable_list[i].load_state_dict(torch.load(temp_path))
            times_lr_adjusted += 1
            continue

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print(f'best validation accuracy: {round(float(best_acc), 3)}%')
    print(f'test accuracy: {round(float(test_acc), 3)}%')

    # log best validation accuracy and test accuracy
    run.log({'test_acc': test_acc, 'test_acc_top5': test_acc_top5, 'test_loss': test_loss, 'best_val_acc': best_acc}) 


if __name__ == '__main__':
    main()
