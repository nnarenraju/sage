# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Fri Jan 28 19:08:44 2022

__author__      = nnarenraju
__copyright__   = Copyright 2021, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

# IN-BUILT
import torch


""" WRAPPERS """

class LossWrapper:
    def __init__(self, always_apply=True):
        self.always_apply = always_apply
        
    def forward(self, outputs, targets, pe):
        raise NotImplementedError
    
    def __call__(self, outputs, targets, pe):
        if self.always_apply:
            return self.forward(outputs, targets, pe)
        else:
            pass


""" CUSTOM LOSS FUNCTIONS """

class BCEgw_MSEtc(LossWrapper):
    
    def __init__(self, always_apply=True, mse_alpha=0.0):
        super().__init__(always_apply)
        assert mse_alpha >= 0.0
        self.mse_alpha = mse_alpha
        
    def forward(self, outputs, targets, pe):
        # BCE to check whether the signal contains GW or is pure noise
        # MSE to add soft weight to the calculation of correct 'tc'
        # Output and target contain (isGW, tc). This is not a two class problem.
        
        """
        PyTorch - BCEWithLogitsLoss
            >>> target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
            >>> output = torch.full([10, 64], 1.5)  # A prediction (logit)
            >>> pos_weight = torch.ones([64])  # All weights are equal to 1
            >>> criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            >>> criterion(output, target)  # -log(sigmoid(1.5))
            tensor(0.2014)
            
        # Convert 1-class output to 2-class output
        outputs_ = np.array(1.0 - outputs)
        
        """
        
        # Creating loss function with weighted action
        criterion = torch.nn.BCEWithLogitsLoss()
        # criterion = torch.nn.BCELoss(weight=self.pos_weight)
        # criterion = regularised_BCEWithLogitsLoss(dim=1)
        # Loss Topic: Does the given signal contain a GW or is it pure noise?
        BCEgw = criterion(outputs['pred_prob'], targets['gw'])
        
        """ Converting to numpy arrays """
        # detached_outputs = {}
        # detached_targets = {}
        # for key in pe:
        #     detached_outputs[key] = outputs[key].detach().cpu().numpy()
        #     detached_targets[key] = targets[key].detach().cpu().numpy()
        
        """
        MSE - Mean Squared Error Loss
        For the handling of 'tc'
        MSEloss = (alpha / N_batch) * SUMMATION (target_tc - pred_tc)^2 / variance_tc
        """
        prefix = self.mse_alpha
        # Use a variance term if required in the mse loss
        # mse_loss = sum((targets[:,1]-outputs[:,1])**2/np.var(outputs[:,1]))
        MSEpe = 0
        for key in pe:
            MSEpe += prefix * torch.mean((targets[key]-outputs[key])**2)
        
        """ 
        CUSTOM LOSS FUNCTION
        L = BCE(P_0) + alpha * MSE(P_1)
        """
        # print("BCE loss = {} and MSE loss = {}".format(BCEgw, MSEtc))
        
        
        custom_loss = BCEgw + MSEpe # not a leaf variable
        # custom_loss = torch.tensor(custom_loss) # is a leaf variable
        # # Not using the following option leads to the following error
        # # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # # Link: https://discuss.pytorch.org/t/runtimeerror-element-0-of-variables-does-not-require-grad-and-does-not-have-a-grad-fn/11074/7
        # custom_loss.requires_grad = True # can be applied only to leaf variable
        
        return custom_loss


class regularised_BCELoss(torch.nn.BCELoss):
    
    def __init__(self, *args, epsilon=1e-6, dim=None, **kwargs):
        torch.nn.BCELoss.__init__(self, *args, **kwargs)
        assert isinstance(dim, int)
        self.regularization_dim = dim
        self.regularization_A = epsilon
        self.regularization_B = 1. - epsilon*self.regularization_dim
    
    def __call__(self, outputs, targets, pe):
        return self.forward(outputs['pred_prob'], targets)
    
    def forward(self, outputs, targets, *args, **kwargs):
        assert outputs.shape[-1] == self.regularization_dim
        transformed_input = self.regularization_A + self.regularization_B*outputs
        return torch.nn.BCELoss.forward(self, transformed_input, targets, *args, **kwargs)


class regularised_BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    
    def __init__(self, *args, epsilon=1e-6, dim=None, **kwargs):
        torch.nn.BCEWithLogitsLoss.__init__(self, *args, **kwargs)
        assert isinstance(dim, int)
        self.regularization_dim = dim
        self.regularization_A = epsilon
        self.regularization_B = 1. - epsilon*self.regularization_dim
    
    def __call__(self, outputs, targets, pe):
        # We use raw values here as BCEWithLogitsLoss has a Sigmoid wrapper
        raw_outputs = outputs['raw'].reshape(len(outputs['raw']), 1)
        raw_targets = targets.reshape(len(targets), 1)
        return self.forward(raw_outputs, raw_targets)
    
    def forward(self, outputs, targets, *args, **kwargs):
        assert outputs.shape[-1] == self.regularization_dim
        transformed_input = self.regularization_A + self.regularization_B*outputs
        return torch.nn.BCEWithLogitsLoss.forward(self, transformed_input, targets, *args, **kwargs)
    
