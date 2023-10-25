import numpy as np
from .TrainHelpers import TrainHelpers
import torch.nn.functional as F

class MixupTrainHelpers(TrainHelpers):
    def process_data(self, data):
        pos, neg, labels_pos, labels_neg, logit_t_pos, logit_t_neg = data
        self.pos_orig_shape = pos.shape
        self.lam = np.random.beta(1, 1)
        self.pos = pos.cuda()
        self.neg = neg.cuda()
        self.labels_pos = labels_pos.cuda()
        self.labels_neg = labels_neg.cuda()
        self.logit_t_pos = logit_t_pos.cuda()
        self.logit_t_neg = logit_t_neg.cuda()

    def model_forward(self, model, preact):
        return model(self.lam * self.pos + (1 - self.lam) * self.neg, is_feat=True, preact=preact)
    
    def compute_outputs(self, model_s, data, preact=False):
        self.process_data(data)
        _, self.logit_s = self.model_forward(model_s, preact)

    def get_batch_size(self):
        return self.pos_orig_shape[0]
    
    def get_cls_loss(self):
        if self.w_cls == 0:
            return 0
        return self.w_cls * (self.lam * self.crit_cls(self.logit_s, self.labels_pos) + (1 - self.lam) * self.crit_cls(self.logit_s, self.labels_neg))

    """Computes and returns the vanilla KD loss"""
    def get_kd_loss(self):
        if self.w_kd == 0:
            return 0
        return self.w_kd * (self.lam * self.crit_kd(self.logit_s, self.logit_t_pos) + (1 - self.lam) * self.crit_kd(self.logit_s, self.logit_t_neg))

    """Computes and returns the relational KD loss"""
    def get_rel_loss(self):
        if self.w_rel == 0:
            return 0
        logit_s, logit_t_pos, logit_t_neg = self.logit_s, self.logit_t_pos, self.logit_t_neg
        if self.opt.loss_fn == 0:
            logit_t_pos = F.softmax(logit_t_pos, dim=1)
            logit_t_neg = F.softmax(logit_t_neg, dim=1)
        return self.w_rel * (self.lam * self.crit_rel(logit_s, logit_t_pos) + (1 - self.lam) * self.crit_rel(logit_s, logit_t_neg))
