import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, prior):
        super(Loss, self).__init__()
        self.prior = prior
        
    def __call__(self, z, sum_log_det_jacobians):
        log_p = self.prior.log_prob(z)
        return -(log_p + sum_log_det_jacobians).mean()
