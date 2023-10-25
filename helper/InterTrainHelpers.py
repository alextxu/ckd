import torch

class InterTrainHelpers:
    """Sets loss functions and loss component weights"""
    def __init__(self, opt, crit_cls, w_cls, crit_kd, w_kd, crit_rel=None, w_rel=0):
        self.opt = opt
        self.crit_cls = crit_cls
        self.crit_kd = crit_kd
        self.crit_rel = crit_rel
        self.w_cls = w_cls
        self.w_kd = w_kd
        self.w_rel = w_rel
    
    """Stores the data from this training batch on the GPU"""
    def process_data(self, data):
        images, labels, logit_t, cached_input = data
        self.images = images.cuda()
        self.labels = labels.cuda()
        self.logit_t = logit_t.cuda()
        self.cached_input = cached_input.cuda()
    
    """Passes the stored data through the model"""
    def model_forward(self, model, inputs, preact=False):
        return model(inputs, is_feat=True, preact=preact)

    """
    Computes and stores the outputs of the two models from the data.
    Uses process_data and model_forward.
    """
    def compute_outputs(self, model_s, model_t, data, preact=False):
        self.process_data(data)
        self.feat_s, self.logit_s = self.model_forward(model_s, self.images, preact)
        with torch.no_grad():
            self.feat_t, _ = self.model_forward(model_t, self.cached_input, preact)
            self.feat_t = [f.detach() for f in self.feat_t]
        # with torch.no_grad():
        #     feat_t, self.logit_t = self.model_forward(model_t, preact)
        #     self.feat_t = [f.detach() for f in feat_t]
    
    """Computes and returns the classification loss"""
    def get_cls_loss(self):
        if self.w_cls == 0:
            return 0
        return self.w_cls * self.crit_cls(self.logit_s, self.labels)

    """Computes and returns the vanilla KD loss"""
    def get_kd_loss(self):
        if self.w_kd == 0:
            return 0
        return self.w_kd * self.crit_kd(self.logit_s, self.logit_t)

    """Computes and returns the relational KD loss"""
    def get_rel_loss(self):
        if self.w_rel == 0:
            return 0
        return self.w_rel * self.crit_rel(self.logit_s, self.logit_t)
    
    """Computes and returns the total loss"""
    def get_total_loss(self):
        temp1 = self.get_cls_loss()
        temp2 = self.get_kd_loss()
        temp3 = self.get_rel_loss()
        return temp1 + temp2 + temp3
    
    def get_batch_size(self):
        return self.images.size(0)