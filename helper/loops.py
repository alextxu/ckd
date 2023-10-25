from __future__ import print_function, division

import sys
import time
import torch
import os

from .util import AverageMeter, accuracy, get_config_str, adjust_learning_rate


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg

def train_distill(run, helper_object, train_loader, val_loader, test_loader,
                  module_list, trainable_list, criterion_inter, val_criterion,
                  optimizer, opt, stats):
    tot_steps, best_acc, last_improve, test_acc, test_acc_top5, test_loss = stats
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    losses = AverageMeter()

    # simulate an epoch in the original subset for fairness
    eval_every = (int(opt.subset_size * opt.train_val_frac) + opt.batch_size - 1) // opt.batch_size

    end = time.time()
    done = False
    for idx, data in enumerate(train_loader):
        preact = False
        if opt.distill != 'kd':
            helper_object.compute_outputs(model_s, model_t, data, preact)
        else:
            helper_object.compute_outputs(model_s, data, preact)
        loss = helper_object.get_total_loss()

        # other distillation losses with intermediate layers
        if opt.distill == 'kd':
            loss_inter = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](helper_object.feat_s[opt.hint_layer])
            f_t = helper_object.feat_t[opt.hint_layer]
            loss_inter = criterion_inter(f_s, f_t)
        elif opt.distill == 'correlation':
            f_s = module_list[1](helper_object.feat_s[-1])
            f_t = module_list[2](helper_object.feat_t[-1])
            loss_inter = criterion_inter(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = helper_object.feat_s[1:-1]
            g_t = helper_object.feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_inter)]
            loss_inter = sum(loss_group)
        else:
            raise NotImplementedError(opt.distill)

        loss += opt.w_inter * loss_inter

        losses.update(loss.item(), helper_object.get_batch_size())
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                tot_steps // eval_every, idx, len(train_loader), batch_time=batch_time,
                loss=losses))
            sys.stdout.flush()

        # validation
        tot_steps += 1
        if tot_steps % eval_every == 0:
            val_acc, val_acc_top5, val_loss = validate(val_loader, model_s, val_criterion, opt)
            model_s.train()

            print('Validation:')
            print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(val_acc, val_acc_top5))
            run.log({'val_acc': val_acc, 'val_loss': val_loss, 'val_acc_top5': val_acc_top5}, step=tot_steps // eval_every)

            if val_acc > best_acc:
                last_improve = tot_steps
                best_acc = val_acc
                if not os.path.isdir('temp'):
                    os.mkdir('temp')
                for i in range(len(trainable_list)):
                    module = trainable_list[i]
                    
                    temp_path = f"temp/{get_config_str(run.config)}_{str(i)}.pth"
                    torch.save(module.state_dict(), temp_path)
                test_acc, test_acc_top5, test_loss = validate(test_loader, model_s, val_criterion, opt)
                model_s.train()

            if tot_steps // eval_every >= opt.epochs:   # reached max epochs
                done = True
                break
            elif (tot_steps - last_improve) // eval_every >= opt.patience:  # early stopping
                done = True
                break

    stats = tot_steps, best_acc, last_improve, test_acc, test_acc_top5, test_loss
    return losses.avg, stats, done

def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
