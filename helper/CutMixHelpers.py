from .TrainHelpers import TrainHelpers
import numpy as np
import torch
import torch.nn.functional as F


class CutMixHelpers(TrainHelpers):
    def __init__(self, opt, crit_cls, w_cls, crit_kd, w_kd, crit_rel=None, w_rel=0):
        super().__init__(opt, crit_cls, w_cls, crit_kd, w_kd, crit_rel, w_rel)
        if opt.dataset == 'cifar100':
            self.n_cls = 100
        else:
            self.n_cls = 0
        self.opt = opt

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

    # @mst: refer to cutmix official impel:
    # https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py#L279
    def rand_bbox(self, size, lam):
        H, W = size[2], size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def make_one_hot(self, labels, C):  # labels: [N]
        """turn a batch of labels to the one-hot form"""
        labels = labels.unsqueeze(1)  # [N, 1]
        one_hot = torch.zeros(labels.size(0), C).cuda()
        target = one_hot.scatter_(1, labels, 1)
        return target

    def get_batch_size(self):
        return self.pos_orig_shape[0]

    def cutmix(self):
        """Refer to official cutmix impl.: https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py#L234"""
        batch_size, _, h, w = self.pos_orig_shape
        self.lam = np.random.beta(1, 1)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(
            self.pos_orig_shape, self.lam
        )  # self.lam is the area ratio of the remaining part over the original image

        # get new image
        mask = torch.ones((h, w)).cuda()
        mask[bby1:bby2, bbx1:bbx2] = 0
        input_mix = self.pos * mask + self.neg * (1 - mask)

        # adjust lambda to exactly match pixel ratio
        self.lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (h * w)

        # linearly interpolate target and logit_t in the same way
        target_pos_oh = self.make_one_hot(self.labels_pos, self.n_cls)
        target_neg_oh = self.make_one_hot(self.labels_neg, self.n_cls)
        target_mix  = self.lam * target_pos_oh + (1 - self.lam) * target_neg_oh

        return input_mix, target_mix

    def model_forward(self, model, preact):
        input_mix, target_mix = self.cutmix()
        _, logit_s = model(input_mix, is_feat=True, preact=preact)
        return logit_s, target_mix

    def compute_outputs(self, model_s, data, preact=False):
        self.process_data(data)
        logit_s, target_mix = self.model_forward(model_s, preact)
        self.logit_s = logit_s
        self.target_mix = target_mix


    def get_cls_loss(self):
        return self.w_cls * (self.lam * self.crit_cls(self.logit_s, self.labels_pos) + (1 - self.lam) * self.crit_cls(self.logit_s, self.labels_neg))

    def get_kd_loss(self):
        return 0

    def get_rel_loss(self):
        if self.w_rel == 0:
            return 0
        logit_s, logit_t_pos, logit_t_neg = self.logit_s, self.logit_t_pos, self.logit_t_neg
        if self.opt.loss_fn == 0:
            logit_t_pos = F.softmax(logit_t_pos, dim=1)
            logit_t_neg = F.softmax(logit_t_neg, dim=1)

        return self.w_rel * (self.lam * self.crit_rel(logit_s, logit_t_pos) + (1 - self.lam) * self.crit_rel(logit_s, logit_t_neg))