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
    
    def __init__(self, always_apply=True, network_snr_for_noise=False, mse_alpha=0.5, gw_criterion=None,
                 emphasis_threshold=0.7, noise_emphasis=False, signal_emphasis=False, emphasis_alpha=0.5,
                 fp_boundary_loss=False, fn_boundary_loss=False, boundary_loss_alpha=0.5):
        
        super().__init__(always_apply)
        # Assertions on input params
        assert mse_alpha >= 0.0, 'mse_alpha must be greater than 0.0'
        assert emphasis_threshold >= 0.0, 'emphasis_threshold on pred_prob must be greater than 0.0'
        assert emphasis_threshold <= 1.0, 'emphasis_threshold set on pred_prob value and not raw. Set value <= 1.0'
        assert boundary_loss_alpha >= 0.0, 'boundary loss must have alpha value greater than 0.0'
        
        # Set generic params
        self.mse_alpha = mse_alpha
        self.network_snr_for_noise = network_snr_for_noise
        if gw_criterion == None:
            self.gw_criterion = torch.nn.BCEWithLogitsLoss()
        
        # Extra loss params
        self.emphasis_threshold = emphasis_threshold
        self.noise_emphasis = noise_emphasis
        self.signal_emphasis = signal_emphasis
        self.emphasis_alpha = emphasis_alpha
        self.fp_boundary_loss = fp_boundary_loss
        self.fn_boundary_loss = fn_boundary_loss
        self.boundary_loss_alpha = boundary_loss_alpha
        
    def forward(self, outputs, targets, pe):
        # BCE to check whether the signal contains GW or is pure noise
        # MSE for calculation of correct 'tc'
        custom_loss = {}
        ## Criterions for GW prediction probabilities
        losses = ['regularised_BCELoss', 'regularised_BCEWithLogitsLoss']
        if self.gw_criterion.__class__.__name__ not in losses:
            if isinstance(self.gw_criterion, torch.nn.BCEWithLogitsLoss):
                # Check for emphasis loss
                if not self.noise_emphasis and not self.signal_emphasis:
                    emphasis_loss = 0.0
                else:
                    # First order mask must be made using pred_prob values as raw is unbounded
                    mask = torch.ge(outputs['pred_prob'], self.emphasis_threshold)
                    # Get all values above the emphasis threshold for outputs and targets
                    masked_target = torch.masked_select(targets['gw'], mask)
                    masked_output = torch.masked_select(outputs['raw'], mask)
                    # Create a second order mask that isolates on signal or noise
                    if self.noise_emphasis and self.signal_emphasis:
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
                
                # Check for fp or fn boundary loss
                boundary_loss = 0.0
                if self.fp_boundary_loss:
                    noise_mask = torch.eq(targets['gw'], 0.0)
                    # Get the outputs of noise
                    masked_noise = torch.masked_select(outputs['pred_prob'], noise_mask)
                    # Calculate the difference between 1.0 and noise stat
                    # 1.0 - max_noise_stat must be positive and as large as possible
                    max_noise_stat = torch.max(masked_noise)
                    # FP boundary loss (loss will be high for high max noise stat)
                    fp_boundary_loss = self.boundary_loss_alpha * max_noise_stat
                    boundary_loss += fp_boundary_loss
                
                if self.fn_boundary_loss:
                    signal_mask = torch.eq(targets['gw'], 1.0)
                    # Get the outputs of noise
                    masked_signal = torch.masked_select(outputs['pred_prob'], signal_mask)
                    # Calculate the difference between min value of signal stat and 0.0
                    # min_signal_stat must be positive and as large as possible
                    min_signal_stat = torch.min(masked_signal)
                    # FP boundary loss
                    fn_boundary_loss = self.boundary_loss_alpha * (1.0 - min_signal_stat)
                    boundary_loss += fn_boundary_loss
                
                # Total loss for GW detection
                BCEgw = self.gw_criterion(outputs['raw'], targets['gw']) + emphasis_loss + boundary_loss

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
            assert -1 not in masked_target, 'Found invalid value (-1) in PE target!'
            # Calculating the individual PE MSE loss
            if key == 'norm_snr' and self.network_snr_for_noise:
                # All samples are included in the prediction of SNR (noise and signals)
                pe_loss = self.mse_alpha * torch.mean((targets[key]-outputs[key])**2)
            else:
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
    
