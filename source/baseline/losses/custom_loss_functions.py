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
    
    def __init__(self, always_apply=True, mse_alpha=0.5, gw_criterion=None):
        super().__init__(always_apply)
        assert mse_alpha >= 0.0
        self.mse_alpha = mse_alpha
        if gw_criterion == None:
            self.gw_criterion = torch.nn.BCEWithLogitsLoss()
        
    def forward(self, outputs, targets, pe):
        # BCE to check whether the signal contains GW or is pure noise
        # MSE for calculation of correct 'tc'
        
        ## Criterions for GW prediction probabilities
        BCEgw = self.gw_criterion(outputs, targets)
        
        """
        MSE - Mean Squared Error Loss
        For the handling of 'tc'
        MSEloss = (alpha / N_batch) * SUMMATION (target_tc - pred_tc)^2 / variance_tc
        """
        MSEpe = 0
        for key in pe:
            MSEpe += self.mse_alpha * torch.mean((targets[key]-outputs[key])**2)
        
        """ 
        CUSTOM LOSS FUNCTION
        L = BCE(P_0) + alpha * MSE(P_1)
        """
        custom_loss = BCEgw + MSEpe
        
        return custom_loss


class regularised_BCELoss(torch.nn.BCELoss):
    
    def __init__(self, *args, epsilon=1e-6, dim=None, **kwargs):
        torch.nn.BCELoss.__init__(self, *args, **kwargs)
        assert isinstance(dim, int)
        self.regularization_dim = dim
        self.regularization_A = epsilon
        self.regularization_B = 1. - epsilon*self.regularization_dim
    
    def __call__(self, outputs, targets, pe):
        raw_outputs = outputs['pred_prob'].reshape(len(outputs['pred_prob']), 1)
        raw_targets = targets['gw'].reshape(len(targets['gw']), 1)
        return self.forward(raw_outputs, raw_targets)
    
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
        raw_targets = targets['gw'].reshape(len(targets['gw']), 1)
        return self.forward(raw_outputs, raw_targets)
    
    def forward(self, outputs, targets, *args, **kwargs):
        assert outputs.shape[-1] == self.regularization_dim
        transformed_input = self.regularization_A + self.regularization_B*outputs
        return torch.nn.BCEWithLogitsLoss.forward(self, transformed_input, targets, *args, **kwargs)
    
