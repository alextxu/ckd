import numpy as np
from .TrainHelpers import TrainHelpers
import torch.nn.functional as F

class MixupThreeTrainHelpers(TrainHelpers):
    def process_data(self, data):
        img1, img2, img3, lbl1, lbl2, lbl3, tlog1, tlog2, tlog3 = data
        self.orig_shape1 = img1.shape
        self.lam = np.random.beta(1, 1, 3)
        self.lam /= np.sum(self.lam)
        self.lam = self.lam.tolist()
        self.img1, self.img2, self.img3 = img1.cuda(), img2.cuda(), img3.cuda()
        self.lbl1, self.lbl2, self.lbl3 = lbl1.cuda(), lbl2.cuda(), lbl3.cuda()
        self.tlog1, self.tlog2, self.tlog3 = tlog1.cuda(), tlog2.cuda(), tlog3.cuda()

    def model_forward(self, model, preact):
        mixed_img = self.lam[0] * self.img1 + self.lam[1] * self.img2 + self.lam[2] * self.img3
        return model(mixed_img, is_feat=True, preact=preact)
    
    def compute_outputs(self, model_s, data, preact=False):
        self.process_data(data)
        self.feat_s, self.logit_s = self.model_forward(model_s, preact)

    def get_batch_size(self):
        return self.orig_shape1[0]
    
    def get_cls_loss(self):
        if self.w_cls == 0:
            return 0
        temp1 = self.lam[0] * self.crit_cls(self.logit_s, self.lbl1)
        temp2 = self.lam[1] * self.crit_cls(self.logit_s, self.lbl2)
        temp3 = self.lam[2] * self.crit_cls(self.logit_s, self.lbl3)
        return self.w_cls * (temp1 + temp2 + temp3)

    def get_kd_loss(self):
        if self.w_kd == 0:
            return 0
        temp1 = self.lam[0] * self.crit_kd(self.logit_s, self.tlog1)
        temp2 = self.lam[1] * self.crit_kd(self.logit_s, self.tlog2)
        temp3 = self.lam[2] * self.crit_kd(self.logit_s, self.tlog3)
        return self.w_kd * (temp1 + temp2 + temp3)

    def get_rel_loss(self):
        if self.w_rel == 0:
            return 0
        tlog1, tlog2, tlog3 = self.tlog1, self.tlog2, self.tlog3
        if self.opt.loss_fn == 0:
            tlog1 = F.softmax(tlog1, dim=1)
            tlog2 = F.softmax(tlog2, dim=1)
            tlog3 = F.softmax(tlog3, dim=1)
        temp1 = self.lam[0] * self.crit_rel(self.logit_s, tlog1)
        temp2 = self.lam[1] * self.crit_rel(self.logit_s, tlog2)
        temp3 = self.lam[2] * self.crit_rel(self.logit_s, tlog3)
        return self.w_rel * (temp1 + temp2 + temp3)

