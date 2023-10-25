from .TrainHelpers import TrainHelpers

class CRDTrainHelpers(TrainHelpers):
    def process_data(self, data):
        images, labels, logit_t, index, contrast_idx = data
        self.images = images.cuda()
        self.labels = labels.cuda()
        self.logit_t = logit_t.cuda()
        self.index = index.cuda()
        self.contrast_idx = contrast_idx.cuda()
    
    def get_rel_loss(self):
        if self.w_rel == 0:
            return 0
        return self.w_rel * self.crit_rel(self.logit_s, self.logit_t, self.index, self.contrast_idx)