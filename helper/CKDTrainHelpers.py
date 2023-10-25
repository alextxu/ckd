import torch
import numpy as np
from .TrainHelpers import TrainHelpers

class CKDTrainHelpers(TrainHelpers):
    def mix_func(self, x, y):
        if self.opt.relational == 'ckd':
            return x - y
        elif self.opt.relational == 'ckd_add':
            return 0.5 * x + 0.5 * y
        elif self.opt.relational == 'ckd_mixup':
            return self.lam * x + (1 - self.lam) * y
        else:
            raise NotImplementedError('unknown mix func')

    def process_data(self, data):
        pos, neg, label_pos, label_neg, logit_t_pos, logit_t_neg = data
        if self.opt.relational == 'ckd_mixup':
            self.lam = np.random.beta(1, 1)
        self.pos_orig_shape = pos.shape
        self.neg_orig_shape = neg.shape
        pos = pos.reshape((pos.shape[0] * pos.shape[1],) + pos.shape[2:])
        neg = neg.reshape((neg.shape[0] * neg.shape[1],) + neg.shape[2:])
        self.pos = pos.cuda()
        self.neg = neg.cuda()
        self.labels = self.mix_func(label_pos.cuda(), label_neg.cuda())
        self.logit_t = self.mix_func(logit_t_pos.cuda(), logit_t_neg.cuda())

    def model_forward(self, model, preact=False):
        feat_pos, logit_pos = model(self.pos, is_feat=True, preact=preact)
        feat_neg, logit_neg = model(self.neg, is_feat=True, preact=preact)
        logit_pos = logit_pos.reshape(self.pos_orig_shape[0], self.pos_orig_shape[1], -1)
        logit_pos = torch.mean(logit_pos, axis=1)
        logit_neg = logit_neg.reshape(self.neg_orig_shape[0], self.neg_orig_shape[1], -1)
        logit_neg = torch.mean(logit_neg, axis=1)
        logit_mixed = self.mix_func(logit_pos, logit_neg)

        avg_feat_dif = []
        for i in range(len(feat_pos)):
            f_pos = feat_pos[i]
            f_neg = feat_neg[i]
            f_pos_shape = f_pos.size()[1:]
            f_neg_shape = f_neg.size()[1:]
            f_pos = f_pos.reshape((self.pos_orig_shape[0], self.pos_orig_shape[1]) + f_pos_shape)
            f_pos = torch.mean(f_pos, axis=1)
            f_neg = f_neg.reshape((self.neg_orig_shape[0], self.neg_orig_shape[1]) + f_neg_shape)
            f_neg = torch.mean(f_neg, axis=1)
            avg_feat_dif.append(self.mix_func(f_pos, f_neg))

        return avg_feat_dif, logit_mixed
        
    def get_batch_size(self):
        return self.pos_orig_shape[0]