import torch
import torch.nn as nn

class RealNVP(nn.Module):
    def __init__(self):
        super(RealNVP, self).__init__()
        layers = []
        
        self.num_scales = 2
        
   
        self.coupling_1 = CouplingLayer("01", 2)
        self.coupling_2 = CouplingLayer("10", 2)
        self.coupling_3 = CouplingLayer("01", 2)
        self.coupling_4 = CouplingLayer("10", 2)
        self.coupling_5 = CouplingLayer("01", 2)
        self.coupling_6 = CouplingLayer("10", 2)
        
        
        self.layers = nn.Sequential(*layers)       
            
    def forward(self, x, reverse=False):
        if not reverse:
            sum_log_det_jacobians = x.new_zeros(x.size(0))
            
            z, log_det_jacobians = self.coupling_1(x, reverse)
            sum_log_det_jacobians += log_det_jacobians
            
            z, log_det_jacobians = self.coupling_2(z, reverse)
            sum_log_det_jacobians += log_det_jacobians
            
            z, log_det_jacobians = self.coupling_3(z, reverse)
            sum_log_det_jacobians += log_det_jacobians
            
            z, log_det_jacobians = self.coupling_4(z, reverse)
            sum_log_det_jacobians += log_det_jacobians
            
            z, log_det_jacobians = self.coupling_5(z, reverse)
            sum_log_det_jacobians += log_det_jacobians
            
            z, log_det_jacobians = self.coupling_6(z, reverse)
            sum_log_det_jacobians += log_det_jacobians

            return z, sum_log_det_jacobians
        else:
            output = self.coupling_6(x, reverse)
            output = self.coupling_5(output, reverse)
            output = self.coupling_4(output, reverse)
            output = self.coupling_3(output, reverse)
            output = self.coupling_2(output, reverse)
            output = self.coupling_1(output, reverse)
            
            return output

class CouplingLayer(nn.Module):
    def __init__(self, mask_type, input_channel):
        super(CouplingLayer, self).__init__()
        self.function_s_t = Function_s_t(input_channel)
        self.mask_type = mask_type
            
    def get_mask(self, num):
        ############################################
        # mask function 구현
        ############################################
        if '01' in self.mask_type:
            mask = torch.tensor([[0.0, 1.0]])
        else:
            mask = torch.tensor([[1.0, 0.0]])
            
        return mask
        
    def forward(self, x, reverse=False):
        
        if not reverse:
            # get mask
            mask = self.get_mask(self.mask_type)
            
            # masked half of x
            x1 = x * mask
            s,t = self.function_s_t(x1, mask)

            # z_1:d = x_1:d
            # z_d+1:D = exp(s(x_1:d)) * x_d+1:D + m(x_1:d)  
            y = x1 + ((-mask+1.0) * (x*torch.exp(s)+t))

            # calculation of jacobians
            log_det_jacobian = torch.sum(s, 1)
            
            return y, log_det_jacobian
        else:
            # get mask
            mask = self.get_mask(self.mask_type)

            # masked half of y
            x1 = x * mask
            s,t = self.function_s_t(x1, mask)
            
            # x_1:d = z_1:d
            # x_d+1:D = z_d+1:D - m(z_1:d) * exp(-s(z_1:d))
            y = x1 + (-mask+1.0) * ((x-t) * torch.exp(-s))

            return y

class Function_s_t(nn.Module):
    ############################################
    # scale, translation function 구현
    ############################################
    def __init__(self, input_channel, num_blocks=1, channel=256):
        super(Function_s_t, self).__init__()
        self.input_channel = input_channel
        layers = []

        layers += [
            nn.Linear(input_channel, channel),
            nn.LeakyReLU(),
            nn.Linear(channel, channel),
            nn.LeakyReLU(),
            nn.Linear(channel, input_channel*2)]
        
        self.model = nn.Sequential(*layers)
        self.w_scale = torch.rand(1, requires_grad=True)
    
    def forward(self, x, mask):
        x = self.model(x)
        
        #######################################
        # scale function : first half dimension
        # translation function : second half dimension
        #######################################
        s = x[:,:self.input_channel] * (-mask+1)
        t = x[:,self.input_channel:] * (-mask+1)
        
        s = nn.Tanh()(s)
        
        return s, t


