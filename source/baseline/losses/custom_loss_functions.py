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
    
    def __init__(self, always_apply=True, gw_criterion=None, mse_alpha=0.2,
                 emphasis_threshold=0.7, emphasis_operator=None, 
                 noise_emphasis=False, signal_emphasis=False, 
                 emphasis_alpha=0.2, emphasis_type='pred_prob', 
                 snr_loss=False, snr_low=8.0, snr_high=15.0, snr_alpha=0.2,
                 mchirp_loss=False, mchirp_operator=None, mchirp_thresh=0.7, mchirp_alpha=0.2):
        
        super().__init__(always_apply)
        ## Assertions on input params
        # MSE Loss is always ON with PE
        assert mse_alpha >= 0.0 and mse_alpha <= 1.0, 'mse_alpha must be within [0, 1]'
        # Emphasis loss
        if noise_emphasis or signal_emphasis:
            if emphasis_type == 'pred_prob':
                assert emphasis_threshold >= 0.0 and emphasis_threshold <= 1.0, 'When using pred_prob: 0.0<=thresh<=1.0'
                assert emphasis_alpha >= 0.0 and emphasis_alpha <=1.0, 'emphasis_alpha must be within [0, 1]'
        # SNR loss
        if snr_loss:
            assert snr_alpha >= 0.0 and snr_alpha <=1.0, 'snr_alpha must be within [0, 1]'
        # Mchirp loss
        if mchirp_loss:
            assert mchirp_alpha >= 0.0 and mchirp_alpha <=1.0, 'mchirp_alpha must be within [0, 1]'
        
        # Set generic params
        self.mse_alpha = mse_alpha
        if gw_criterion == None:
            self.gw_criterion = torch.nn.BCEWithLogitsLoss()
        
        ### Extra loss params
        # Emphasis Loss
        self.emphasis_threshold = emphasis_threshold
        self.noise_emphasis = noise_emphasis
        self.signal_emphasis = signal_emphasis
        self.emphasis_alpha = emphasis_alpha
        self.emphasis_type = emphasis_type
        self.emphasis_operator = emphasis_operator
        # SNR Loss
        self.snr_loss = snr_loss
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.snr_alpha = snr_alpha
        # Mchirp loss
        self.mchirp_loss = mchirp_loss
        self.mchirp_thresh = mchirp_thresh
        self.mchirp_alpha = mchirp_alpha
        self.mchirp_operator = mchirp_operator

        
    def forward(self, outputs, targets, pe):
        # BCE to check whether the signal contains GW or is pure noise
        # MSE for calculation of correct 'tc'
        custom_loss = {}
        ## Criterions for GW prediction probabilities
        losses = ['regularised_BCELoss', 'regularised_BCEWithLogitsLoss']
        if self.gw_criterion.__class__.__name__ not in losses:
            if isinstance(self.gw_criterion, torch.nn.BCEWithLogitsLoss):
                
                """ Loss Function Bits and Bobs """
                ### Check for emphasis loss
                if not self.noise_emphasis and not self.signal_emphasis:
                    emphasis_loss = 0.0
                else:
                    # First order mask must be made using pred_prob values as raw is unbounded
                    if self.emphasis_type == 'pred_prob':
                        _check = outputs['pred_prob']
                    elif self.emphasis_type == 'raw':
                        _check = outputs['raw']
                    else:
                        raise ValueError('Emphasis type in custom loss function is invalid!')
                    
                    mask = self.emphasis_operator(_check, self.emphasis_threshold)
                    # Get all values above the emphasis threshold for outputs and targets
                    masked_target = torch.masked_select(targets['gw'], mask)
                    masked_output = torch.masked_select(outputs['raw'], mask)
                    # Create a second order mask that isolates on signal or noise
                    if self.noise_emphasis and self.signal_emphasis:
                        if len(masked_target) == 0:
                            emphasis_loss = 0.0
                        else:
                            emphasis_loss = self.gw_criterion(masked_output, masked_target)
                    elif self.noise_emphasis or self.signal_emphasis:
                        if self.noise_emphasis:
                            emphasise_on = 0.0
                        elif self.signal_emphasis:
                            emphasise_on = 1.0
                        
                        # Create a second order mask to emphasise on noise or signal
                        second_order_mask = torch.eq(masked_target, emphasise_on)
                        second_order_masked_target = torch.masked_select(masked_target, second_order_mask)
                        second_order_masked_output = torch.masked_select(masked_output, second_order_mask)
                        ## Emphasis loss
                        # Sanity check (if no noise values above threshold)
                        if len(second_order_masked_target) == 0:
                            emphasis_loss = 0.0
                        else:
                            emphasis_loss = self.gw_criterion(second_order_masked_output, second_order_masked_target)
                            emphasis_loss = self.emphasis_alpha * emphasis_loss
                
                ### Signals that have SNR within the given range [snr_low, snr_high]
                snr_loss = 0.0
                if self.snr_loss:
                    # First order mask for SNR lower limit
                    snr_lmask = torch.gt(targets['snr'], self.snr_low)
                    snr_masked_output = torch.masked_select(outputs['raw'], snr_lmask)
                    snr_masked_target = torch.masked_select(targets['gw'], snr_lmask)
                    # Second order mask for SNR upper limit
                    second_order_snr_lmask = torch.lt(snr_masked_target, self.snr_high)
                    second_order_snr_masked_target = torch.masked_select(snr_masked_target, second_order_snr_lmask)
                    second_order_snr_masked_output = torch.masked_select(snr_masked_output, second_order_snr_lmask)
                    # Calculating SNR loss
                    snr_loss = self.gw_criterion(second_order_snr_masked_output, second_order_snr_masked_target)
                    snr_loss = self.snr_alpha * snr_loss
                
                ### mchirp loss
                mchirp_loss = 0.0
                if self.mchirp_loss:
                    mc_mask = self.mchirp_operator(targets['norm_mchirp'], self.mchirp_thresh)
                    # Sanity check (if no chirp masses above/below threshold)
                    mc_masked_output = torch.masked_select(outputs['raw'], mc_mask)
                    mc_masked_target = torch.masked_select(targets['gw'], mc_mask)
                    if len(mc_masked_target) != 0:
                        mchirp_loss = self.gw_criterion(mc_masked_output, mc_masked_target)
                        mchirp_loss = self.mchirp_alpha * mchirp_loss
                
                
                """ Total loss for GW detection """
                bits_and_bobs_loss = emphasis_loss + snr_loss + mchirp_loss
                BCEgw = self.gw_criterion(outputs['raw'], targets['gw']) + bits_and_bobs_loss

            elif isinstance(self.gw_criterion, torch.nn.BCELoss):
                if self.noise_emphasis or self.signal_emphasis:
                    raise NotImplementedError('Emphasis loss not implemented for BCELoss')
                BCEgw = self.gw_criterion(outputs['pred_prob'], targets['gw'])
        
        else:
            if self.noise_emphasis or self.signal_emphasis:
                raise NotImplementedError('Emphasis loss not implemented for regularised losses')
            loss = self.gw_criterion(outputs, targets)
            BCEgw = loss['total_loss']
        
        custom_loss['gw'] = BCEgw
        
        
        """
        MSE - Mean Squared Error Loss
        For the handling of 'tc'
        MSEloss = (alpha / N_batch) * SUMMATION (target_tc - pred_tc)^2 / variance_tc
        """
        MSEpe = 0
        for key in pe:
            # Get a masked loss calculation for parameter estimation
            # Ignore all targets corresponding to pure noise samples
            mask = torch.ge(targets[key], 0.0)
            masked_target = torch.masked_select(targets[key], mask)
            masked_output = torch.masked_select(outputs[key], mask)
            assert -1 not in masked_target, 'Found invalid value (-1) in PE target. Noise sample may have leaked!'
            # Calculating the individual PE MSE loss
            pe_loss = self.mse_alpha * torch.mean((masked_target-masked_output)**2)
            # Store losses
            custom_loss[key] = pe_loss
            MSEpe += pe_loss
        
        
        """ 
        CUSTOM LOSS FUNCTION
        L = BCE(P_0) + alpha * MSE(P_1)
        """
        custom_loss['total_loss'] = BCEgw + MSEpe
        
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
        custom_loss = {}
        custom_loss['total_loss'] = torch.nn.BCELoss.forward(self, transformed_input, targets, *args, **kwargs)
        custom_loss['gw'] = custom_loss['total_loss']
        return custom_loss



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
        custom_loss = {}
        custom_loss['total_loss'] = torch.nn.BCEWithLogitsLoss.forward(self, transformed_input, targets, *args, **kwargs)
        custom_loss['gw'] = custom_loss['total_loss']
        return custom_loss
    
