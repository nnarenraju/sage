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
import numpy as np


""" WRAPPERS """

class LossWrapper:
    def __init__(self, always_apply=False):
        self.always_apply = always_apply
        
    def apply(self, outputs, targets):
        raise NotImplementedError
    
    def __call__(self, outputs, targets):
        if self.always_apply:
            return self.apply(outputs, targets)
        else:
            pass


""" CUSTOM LOSS FUNCTIONS """

class BCEgw_MSEtc(LossWrapper):
    
    def __init__(self, always_apply=True, class_weight=None, mse_alpha=1.0):
        super().__init__(always_apply)
        assert mse_alpha >= 0.0
        self.mse_alpha = mse_alpha
        self.pos_weight = class_weight
        
    def apply(self, outputs, targets):
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
        if not self.pos_weight:
            # Change to '2' if using two class outputs
            self.pos_weight = torch.ones([1])
        
        # Creating loss function with weighted action
        # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        criterion = torch.nn.BCELoss(weight=self.pos_weight)
        # Loss Topic: Does the given signal contain a GW or is it pure noise?
        BCEgw = criterion(outputs[:,0].to(dtype=torch.float32), targets[:,0].to(dtype=torch.float32))
        
        """ Converting to numpy arrays """
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        """
        MSE - Mean Squared Error Loss
        For the handling of 'tc'
        MSEloss = (alpha / N_batch) * SUMMATION (target_tc - pred_tc)^2 / variance_tc
        """
        prefix = (self.mse_alpha/outputs.shape[0])
        # mse_loss = sum((targets[:,1]-outputs[:,1])**2/np.var(outputs[:,1]))
        mse_loss = sum((targets[:,1]-outputs[:,1])**2)
        MSEtc = prefix * mse_loss
        MSEtc = torch.tensor(MSEtc, dtype=torch.float32)
        
        """ 
        CUSTOM LOSS FUNCTION
        L = BCE(P_0) + alpha * MSE(P_1)
        """
        print("BCE loss = {} and MSE loss = {}".format(BCEgw, MSEtc))
        
        custom_loss = BCEgw + MSEtc # not a leaf variable
        custom_loss = torch.tensor(custom_loss) # is a leaf variable
        # Not using the following option leads to the following error
        # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # Link: https://discuss.pytorch.org/t/runtimeerror-element-0-of-variables-does-not-require-grad-and-does-not-have-a-grad-fn/11074/7
        custom_loss.requires_grad = True # can be applied only to leaf variable
        
        return custom_loss




""" FOR DIRECT USE ONLY, DO *NOT* USE WITH PYTORCH LIGHTNING """

class reg_BCELoss(torch.nn.BCELoss):
    
    def __init__(self, *args, epsilon=1e-6, dim=None, **kwargs):
        torch.nn.BCELoss.__init__(self, *args, **kwargs)
        assert isinstance(dim, int)
        self.regularization_dim = dim
        self.regularization_A = epsilon
        self.regularization_B = 1. - epsilon*self.regularization_dim
        
    def forward(self, output, target, *args, **kwargs):
        assert output.shape[-1]==self.regularization_dim
        transformed_input = self.regularization_A + self.regularization_B*output
        return torch.nn.BCELoss.forward(self, transformed_input, target, *args, **kwargs)
