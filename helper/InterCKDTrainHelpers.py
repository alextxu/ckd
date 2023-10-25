import torch
import numpy as np
from .InterTrainHelpers import InterTrainHelpers

class InterCKDTrainHelpers(InterTrainHelpers):
    def process_data(self, data):
        pos, neg, label_pos, label_neg, logit_t_pos, logit_t_neg, cached_pos, cached_neg = data
        self.pos_orig_shape = pos.shape
        self.neg_orig_shape = neg.shape
        pos = pos.reshape((pos.shape[0] * pos.shape[1],) + pos.shape[2:])
        neg = neg.reshape((neg.shape[0] * neg.shape[1],) + neg.shape[2:])
        cached_pos = cached_pos.reshape((cached_pos.shape[0] * cached_pos.shape[1],) + cached_pos.shape[2:])
        cached_neg = cached_neg.reshape((cached_neg.shape[0] * cached_neg.shape[1],) + cached_neg.shape[2:])
        self.pos = pos.cuda()
        self.neg = neg.cuda()
        labels = label_pos - label_neg
        logit_t = logit_t_pos - logit_t_neg
        self.labels = labels.cuda()
        self.logit_t = logit_t.cuda()
        self.cached_pos = cached_pos.cuda()
        self.cached_neg = cached_neg.cuda()

    def model_forward(self, model, inputs, preact=False):
        pos, neg = inputs
        feat_pos, logit_pos = model(pos, is_feat=True, preact=preact)
        feat_neg, logit_neg = model(neg, is_feat=True, preact=preact)
        logit_pos = logit_pos.reshape(self.pos_orig_shape[0], self.pos_orig_shape[1], -1)
        logit_pos = torch.mean(logit_pos, axis=1)
        logit_neg = logit_neg.reshape(self.neg_orig_shape[0], self.neg_orig_shape[1], -1)
        logit_neg = torch.mean(logit_neg, axis=1)
        logit_mixed = logit_pos - logit_neg

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
            avg_feat_dif.append(f_pos - f_neg)

        return avg_feat_dif, logit_mixed
    
    def compute_outputs(self, model_s, model_t, data, preact=False):
        self.process_data(data)
        self.feat_s, self.logit_s = self.model_forward(model_s, (self.pos, self.neg), preact)
        with torch.no_grad():
            self.feat_t, _ = self.model_forward(model_t, (self.cached_pos, self.cached_neg), preact)
            self.feat_t = [f.detach() for f in self.feat_t]
        
    def get_batch_size(self):
        return self.pos_orig_shape[0]