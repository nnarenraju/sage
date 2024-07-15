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
    def __init__(self, always_apply=True):
        self.always_apply = always_apply
        
    def forward(self, outputs, targets, source_params, pe):
        raise NotImplementedError
    
    def __call__(self, outputs, targets, source_params, pe):
        if self.always_apply:
            return self.forward(outputs, targets, source_params, pe)
        else:
            pass


""" CUSTOM LOSS FUNCTIONS """

class BCEWithPEregLoss(LossWrapper):
    
    def __init__(self, always_apply=True, gw_criterion=None, 
                 weighted_bce_loss=False, mse_alpha=0.2,
                 emphasis_type='raw',
                 noise_emphasis=False, noise_conditions=[('min_noise', 'max_noise', 0.5),], 
                 signal_emphasis=False, signal_conditions=[(0.0, 'max_noise', 1.0),],
                 snr_loss=False, snr_conditions=[(5.0, 10.0, 0.2),],
                 mchirp_loss=False, mchirp_conditions=[(25.0, 45.0, 0.2),],
                 dchirp_conditions=None,
                 variance_loss=False):
        
        super().__init__(always_apply)

        # Prelim
        self.keyword_limits = {'min_noise':None, 'max_noise':None, 
                               'min_signal':None, 'max_signal':None}
        ## Assertions on input params
        # MSE Loss is always ON with PE
        assert mse_alpha >= 0.0, 'mse_alpha must be greater than or equal to 0.0'
        # Conditional loss
        if noise_emphasis or signal_emphasis or snr_loss or mchirp_loss:
            for condition in signal_conditions+noise_conditions+snr_conditions+mchirp_conditions:
                # Sanity check for keyword conditions
                assert condition[0] in self.keyword_limits.keys() if isinstance(condition[0], str) else True, 'condition[0] should be one of {}'.format(self.keyword_limits.keys())
                assert condition[1] in self.keyword_limits.keys() if isinstance(condition[1], str) else True, 'condition[1] should be one of {}'.format(self.keyword_limits.keys())
                # Emphasis alpha
                assert condition[2] >= 0.0, 'Emphasis alpha cannot be negative. Use a loss function with negative output if intentional.'
                # Emphasis thresholds
                if not isinstance(condition[0], str) and not isinstance(condition[1], str):
                    assert condition[0] <= condition[1], 'Lower threshold > upper threshold. Check emphasis_thresholds in given condition!' 
                    if emphasis_type=='pred_prob' and (signal_emphasis or noise_emphasis):
                        assert 1.0 <= condition[0] >=0.0, 'With emphasis_type=pred_prob, conditions must be within [0.0, 1.0]'
                        assert 1.0 <= condition[1] >=0.0, 'With emphasis_type=pred_prob, conditions must be within [0.0, 1.0]'
        
        # Set generic params
        self.mse_alpha = mse_alpha
        if gw_criterion == None:
            self.gw_criterion = torch.nn.BCEWithLogitsLoss()
        
        ### Extra loss params
        # Emphasis Loss
        self.noise_emphasis = noise_emphasis
        self.signal_emphasis = signal_emphasis
        self.emphasis_type = emphasis_type
        self.noise_conditions = noise_conditions
        self.signal_conditions = signal_conditions
        # SNR Loss
        self.snr_loss = snr_loss
        self.snr_conditions = snr_conditions
        # Mchirp loss
        self.mchirp_loss = mchirp_loss
        self.mchirp_conditions = mchirp_conditions

        self.dchirp_conditions = dchirp_conditions

        # Weighted BCE Loss
        self.weighted_bce_loss = weighted_bce_loss
        # Variance loss
        self.variance_loss = variance_loss

    
    def __str__(self):
        # Display details of loss function
        str = "Loss function = {}".format(self.gw_criterion.__class__.__name__)
        return str
    
    def apply_condition(self, condition, outputs, targets, override_mask_data=None):
        ## Applies gw_criterion to conditional result
        if override_mask_data != None:
            mask_data = override_mask_data
        else:
            mask_data = outputs
        # Assign keyword limits to conditions
        lcondition = self.keyword_limits[condition[0]] if condition[0] in self.keyword_limits.keys() else condition[0]
        ucondition = self.keyword_limits[condition[1]] if condition[1] in self.keyword_limits.keys() else condition[1]
        # Lower limit
        llimit_mask = torch.ge(mask_data, lcondition)
        llimit_mask_data = torch.masked_select(mask_data, llimit_mask)
        llimit_outputs = torch.masked_select(outputs, llimit_mask)
        llimit_targets = torch.masked_select(targets, llimit_mask)
        if len(llimit_targets)!=0:
            return 0.0
        # Upper limit
        ulimit_mask = torch.le(llimit_mask_data, ucondition)
        ulimit_outputs = torch.masked_select(llimit_outputs, ulimit_mask)
        ulimit_targets = torch.masked_select(llimit_targets, ulimit_mask)
        if len(ulimit_targets)!=0:
            return 0.0
        # Get loss
        loss = condition[2] * self.gw_criterion(ulimit_outputs, ulimit_targets)
        return loss

    def forward(self, outputs, targets, source_params, pe):
        # BCE to check whether the signal contains GW or is pure noise
        # MSE for calculation of correct 'tc'
        custom_loss = {}
        
        regularised_losses = ['regularised_BCELoss', 'regularised_BCEWithLogitsLoss']
        if self.gw_criterion.__class__.__name__ not in regularised_losses:
            if isinstance(self.gw_criterion, torch.nn.BCEWithLogitsLoss):
                
                ### COMMON
                if self.signal_emphasis or self.noise_emphasis or self.snr_loss or self.mchirp_loss:
                    # Signal keywords
                    signal_mask = torch.eq(targets['gw'], 1.0)
                    signal_outputs = torch.masked_select(outputs['raw'], signal_mask)
                    signal_targets = torch.masked_select(targets['gw'], signal_mask)
                    # Assign keyword limits
                    self.keyword_limits['min_signal'] = torch.min(signal_outputs)
                    self.keyword_limits['max_signal'] = torch.max(signal_outputs)
                    # Noise keywords
                    noise_mask = torch.eq(targets['gw'], 0.0)
                    noise_outputs = torch.masked_select(outputs['raw'], noise_mask)
                    noise_targets = torch.masked_select(targets['gw'], noise_mask)
                    # Assign keyword limits
                    self.keyword_limits['min_noise'] = torch.min(noise_outputs)
                    self.keyword_limits['max_noise'] = torch.max(noise_outputs)

                ### EMPHASIS LOSS
                if not self.noise_emphasis and not self.signal_emphasis:
                    emphasis_loss = 0.0
                else:
                    # First order mask must be made using pred_prob values as raw is unbounded
                    if self.emphasis_type == 'pred_prob':
                        override = outputs['pred_prob']
                        _check = outputs['raw']
                    elif self.emphasis_type == 'raw':
                        override = None
                        _check = outputs['raw']
                    else:
                        raise ValueError('Emphasis type in custom loss function is invalid!')
                    
                    ## Emphasis conditions
                    emphasis_loss = 0.0
                    # Noise Emphasis
                    if self.noise_emphasis:
                        if len(noise_targets)!=0:
                            for condition in self.noise_conditions:
                                emphasis_loss += self.apply_condition(condition, noise_outputs, noise_targets, override_mask_data=override)
                    # Signal Emphasis 
                    if self.signal_emphasis: 
                        if len(signal_targets)!=0:
                            for condition in self.signal_conditions:
                                emphasis_loss += self.apply_condition(condition, signal_outputs, signal_targets, override_mask_data=override)
                
                ### SNR LOSS
                # Signals that have SNR within the given range [snr_low, snr_high]
                snr_loss = 0.0
                if self.snr_loss and len(signal_targets)!=0:                                    
                    for condition in self.snr_conditions:
                        snr_loss += self.apply_condition(condition, signal_outputs, signal_targets, override_mask_data=targets['snr'])
                
                ### MCHIRP LOSS
                mchirp_loss = 0.0
                if self.mchirp_loss and len(signal_targets)!=0: 
                    for condition in self.mchirp_conditions:
                        mchirp_loss += self.apply_condition(condition, signal_outputs, signal_targets, override_mask_data=source_params['mchirp'])

                
                dchirp_loss = 0.0
                if False:
                    search = source_params['dchirp'].to(device='cuda:1')
                    search = torch.masked_select(search, signal_mask)
                    dc_mask = torch.gt(search, 130.0)
                    # Sanity check (if no chirp masses above threshold)
                    detectable_dc_masked_output = torch.masked_select(outputs['raw'][signal_mask], dc_mask)
                    detectable_dc_masked_targets = torch.masked_select(targets['gw'][signal_mask], dc_mask)
                    if len(detectable_dc_masked_targets) != 0:
                        dchirp_loss = self.gw_criterion(detectable_dc_masked_output, detectable_dc_masked_targets)
                        dchirp_loss = dchirp_loss * 2.0

                """ Loss term for whitening """
                # unet_loss = self.gw_criterion(outputs['whitened'], targets['whitened'])

                """ Total loss for GW detection """
                # Bits and bobs loss from the above set
                bits_and_bobs_loss = emphasis_loss + snr_loss + mchirp_loss + dchirp_loss

                # Weighted BCE loss (if necessary)
                mse_weight = {}
                if self.weighted_bce_loss:
                    bce_loss = 0.0
                    weighted_bce_data = targets['weighted_bce_data']
                    num_params = len(weighted_bce_data.keys())
                    for param, data in weighted_bce_data.items():
                        counts, bins = data
                        counts = np.insert(counts, 0, np.sum(counts), axis=0)
                        assert 0 not in counts, "INVALID: Found a zero count bin in weighted BCE loss."
                        bins = np.insert(bins, 0, -2, axis=0)
                        # Include noise to the weighting as it is one of the classes
                        # Weights are calculated based on the num of samples in input dataset
                        # Add the weights for each parameter specified within weighted bce loss
                        # For noise, we add the same weight for each param given
                        total_count = np.sum(counts)
                        # Calculate weights using counts as 1/nsamples_in_bin
                        # Weight for noise will be 1/(0.5*total_count)
                        # We can convert these weights into effective weights by using percent of samples
                        # Finally, divide total calculated loss by number of parameters (num_params)
                        ## We iterate through each sample and use required weight
                        bin_idx = np.digitize(targets[param], bins)
                        weight = 1./(counts[bin_idx]/total_count)
                        mse_weight['norm_'+param] = [foo for foo in weight if foo != 2.]
                        weighted_criterion = torch.nn.BCEWithLogitsLoss(weight=weight)
                        bce_loss += weighted_criterion(outputs['raw'], targets['gw'])
                else:
                    bce_loss = self.gw_criterion(outputs['raw'], targets['gw'])
                
                # Additional Focal Loss
                # focal_loss = torchvision.ops.sigmoid_focal_loss(outputs['raw'], targets['gw'], reduction='mean')

                # Total BCE Loss
                BCEgw = bce_loss + bits_and_bobs_loss

            elif isinstance(self.gw_criterion, torch.nn.BCELoss):
                if self.noise_emphasis or self.signal_emphasis or self.snr_loss or self.mchirp_loss:
                    raise NotImplementedError('Bits and bobs loss not implemented for BCELoss')
                BCEgw = self.gw_criterion(outputs['pred_prob'], targets['gw'])
        
        else:
            if self.noise_emphasis or self.signal_emphasis or self.snr_loss or self.mchirp_loss:
                raise NotImplementedError('Bits and bobs loss not implemented for regularised losses')
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
            if self.mse_alpha == 0.0:
                pe_loss = torch.tensor(0.0).to(device='cuda:{}'.format(BCEgw.get_device()))
                custom_loss[key] = pe_loss
                MSEpe += pe_loss
                continue

            mask = torch.ge(targets[key], 0.0)
            masked_target = torch.masked_select(targets[key], mask)
            masked_output = torch.masked_select(outputs[key], mask)
            assert -1 not in masked_target, 'Found invalid value (-1) in PE target. Noise sample may have leaked!'
            if len(masked_target) == 0:
                pe_loss = torch.tensor(0.0)
            # Calculating the individual PE MSE loss
            elif self.variance_loss:
                # Variance loss on PE terms
                assert key+"_var" in outputs.keys(), "Variance output not present for {} parameter!".format(key)
                # TODO: Output log(sigma) instead of sigma for HDR.
                param_var = torch.masked_select(outputs[key+'_var'], mask)
                pe_loss = torch.mean(0.5 * ((masked_target-masked_output)/torch.exp(param_var))**2 + param_var)
                pe_loss = self.mse_alpha * pe_loss
            else:
                pe_loss = self.mse_alpha * torch.mean((masked_target-masked_output)**2)

            # Store losses
            if torch.is_tensor(pe_loss) and torch.isnan(pe_loss):
                raise ValueError("PE Loss for {} is nan! val = {}".format(key, pe_loss))
            if not torch.is_tensor(pe_loss):
                if np.isnan(pe_loss):
                    raise ValueError("PE Loss for {} is nan! val = {}".format(key, pe_loss))
            
            custom_loss[key] = pe_loss
            MSEpe += pe_loss

        
        """ 
        CUSTOM LOSS FUNCTION
        L = BCE(P_0) + alpha * MSE(P_1)
        """
        custom_loss['total_loss'] = BCEgw + MSEpe
        
        return custom_loss


class Experimental(LossWrapper):
    
    def __init__(self, always_apply=True, gw_criterion=None, 
                 weighted_bce_loss=False, mse_alpha=0.2,
                 emphasis_type='raw',
                 noise_emphasis=False, noise_conditions=[('min_noise', 'max_noise', 0.5),], 
                 signal_emphasis=False, signal_conditions=[(0.0, 'max_noise', 1.0),],
                 snr_loss=False, snr_conditions=[(5.0, 10.0, 0.2),],
                 mchirp_loss=False, mchirp_conditions=[(25.0, 45.0, 0.2),],
                 dchirp_conditions=None,
                 variance_loss=False):
        
        super().__init__(always_apply)

        # Prelim
        self.keyword_limits = {'min_noise':None, 'max_noise':None, 
                               'min_signal':None, 'max_signal':None}
        ## Assertions on input params
        # MSE Loss is always ON with PE
        assert mse_alpha >= 0.0, 'mse_alpha must be greater than or equal to 0.0'
        # Conditional loss
        if noise_emphasis or signal_emphasis or snr_loss or mchirp_loss:
            for condition in signal_conditions+noise_conditions+snr_conditions+mchirp_conditions:
                # Sanity check for keyword conditions
                assert condition[0] in self.keyword_limits.keys() if isinstance(condition[0], str) else True, 'condition[0] should be one of {}'.format(self.keyword_limits.keys())
                assert condition[1] in self.keyword_limits.keys() if isinstance(condition[1], str) else True, 'condition[1] should be one of {}'.format(self.keyword_limits.keys())
                # Emphasis alpha
                assert condition[2] >= 0.0, 'Emphasis alpha cannot be negative. Use a loss function with negative output if intentional.'
                # Emphasis thresholds
                if not isinstance(condition[0], str) and not isinstance(condition[1], str):
                    assert condition[0] <= condition[1], 'Lower threshold > upper threshold. Check emphasis_thresholds in given condition!' 
                    if emphasis_type=='pred_prob' and (signal_emphasis or noise_emphasis):
                        assert 1.0 <= condition[0] >=0.0, 'With emphasis_type=pred_prob, conditions must be within [0.0, 1.0]'
                        assert 1.0 <= condition[1] >=0.0, 'With emphasis_type=pred_prob, conditions must be within [0.0, 1.0]'
        
        # Set generic params
        self.mse_alpha = mse_alpha
        if gw_criterion == None:
            self.gw_criterion = torch.nn.BCEWithLogitsLoss()
        
        ### Extra loss params
        # Emphasis Loss
        self.noise_emphasis = noise_emphasis
        self.signal_emphasis = signal_emphasis
        self.emphasis_type = emphasis_type
        self.noise_conditions = noise_conditions
        self.signal_conditions = signal_conditions
        # SNR Loss
        self.snr_loss = snr_loss
        self.snr_conditions = snr_conditions
        # Mchirp loss
        self.mchirp_loss = mchirp_loss
        self.mchirp_conditions = mchirp_conditions

        self.dchirp_conditions = dchirp_conditions

        # Weighted BCE Loss
        self.weighted_bce_loss = weighted_bce_loss
        # Variance loss
        self.variance_loss = variance_loss

    
    def __str__(self):
        # Display details of loss function
        str = "Loss function = {}".format(self.gw_criterion.__class__.__name__)
        return str
    
    def apply_condition(self, condition, outputs, targets, override_mask_data=None):
        ## Applies gw_criterion to conditional result
        if override_mask_data != None:
            mask_data = override_mask_data
        else:
            mask_data = outputs
        # Assign keyword limits to conditions
        lcondition = self.keyword_limits[condition[0]] if condition[0] in self.keyword_limits.keys() else condition[0]
        ucondition = self.keyword_limits[condition[1]] if condition[1] in self.keyword_limits.keys() else condition[1]
        # Lower limit
        llimit_mask = torch.ge(mask_data, lcondition)
        llimit_mask_data = torch.masked_select(mask_data, llimit_mask)
        llimit_outputs = torch.masked_select(outputs, llimit_mask)
        llimit_targets = torch.masked_select(targets, llimit_mask)
        if len(llimit_targets)!=0:
            return 0.0
        # Upper limit
        ulimit_mask = torch.le(llimit_mask_data, ucondition)
        ulimit_outputs = torch.masked_select(llimit_outputs, ulimit_mask)
        ulimit_targets = torch.masked_select(llimit_targets, ulimit_mask)
        if len(ulimit_targets)!=0:
            return 0.0
        # Get loss
        loss = condition[2] * self.gw_criterion(ulimit_outputs, ulimit_targets)
        return loss

    def forward(self, outputs, targets, source_params, pe):
        # BCE to check whether the signal contains GW or is pure noise
        # MSE for calculation of correct 'tc'
        custom_loss = {}
        
        regularised_losses = ['regularised_BCELoss', 'regularised_BCEWithLogitsLoss']
        if self.gw_criterion.__class__.__name__ not in regularised_losses:
            if isinstance(self.gw_criterion, torch.nn.BCEWithLogitsLoss):
                
                ### COMMON
                if self.signal_emphasis or self.noise_emphasis or self.snr_loss or self.mchirp_loss:
                    # Signal keywords
                    signal_mask = torch.eq(targets['gw'], 1.0)
                    signal_outputs = torch.masked_select(outputs['raw'], signal_mask)
                    signal_targets = torch.masked_select(targets['gw'], signal_mask)
                    # Assign keyword limits
                    self.keyword_limits['min_signal'] = torch.min(signal_outputs)
                    self.keyword_limits['max_signal'] = torch.max(signal_outputs)
                    # Noise keywords
                    noise_mask = torch.eq(targets['gw'], 0.0)
                    noise_outputs = torch.masked_select(outputs['raw'], noise_mask)
                    noise_targets = torch.masked_select(targets['gw'], noise_mask)
                    # Assign keyword limits
                    self.keyword_limits['min_noise'] = torch.min(noise_outputs)
                    self.keyword_limits['max_noise'] = torch.max(noise_outputs)

                ### EMPHASIS LOSS
                if not self.noise_emphasis and not self.signal_emphasis:
                    emphasis_loss = 0.0
                else:
                    # First order mask must be made using pred_prob values as raw is unbounded
                    if self.emphasis_type == 'pred_prob':
                        override = outputs['pred_prob']
                        _check = outputs['raw']
                    elif self.emphasis_type == 'raw':
                        override = None
                        _check = outputs['raw']
                    else:
                        raise ValueError('Emphasis type in custom loss function is invalid!')
                    
                    ## Emphasis conditions
                    emphasis_loss = 0.0
                    # Noise Emphasis
                    if self.noise_emphasis:
                        if len(noise_targets)!=0:
                            for condition in self.noise_conditions:
                                emphasis_loss += self.apply_condition(condition, noise_outputs, noise_targets, override_mask_data=override)
                    # Signal Emphasis 
                    if self.signal_emphasis: 
                        if len(signal_targets)!=0:
                            for condition in self.signal_conditions:
                                emphasis_loss += self.apply_condition(condition, signal_outputs, signal_targets, override_mask_data=override)
                
                ### SNR LOSS
                # Signals that have SNR within the given range [snr_low, snr_high]
                snr_loss = 0.0
                if self.snr_loss and len(signal_targets)!=0:                                    
                    for condition in self.snr_conditions:
                        snr_loss += self.apply_condition(condition, signal_outputs, signal_targets, override_mask_data=targets['snr'])
                
                ### MCHIRP LOSS
                mchirp_loss = 0.0
                if self.mchirp_loss and len(signal_targets)!=0: 
                    for condition in self.mchirp_conditions:
                        mchirp_loss += self.apply_condition(condition, signal_outputs, signal_targets, override_mask_data=source_params['mchirp'])

                
                dchirp_loss = 0.0
                if False:
                    search = source_params['dchirp'].to(device='cuda:1')
                    search = torch.masked_select(search, signal_mask)
                    dc_mask = torch.gt(search, 130.0)
                    # Sanity check (if no chirp masses above threshold)
                    detectable_dc_masked_output = torch.masked_select(outputs['raw'][signal_mask], dc_mask)
                    detectable_dc_masked_targets = torch.masked_select(targets['gw'][signal_mask], dc_mask)
                    if len(detectable_dc_masked_targets) != 0:
                        dchirp_loss = self.gw_criterion(detectable_dc_masked_output, detectable_dc_masked_targets)
                        dchirp_loss = dchirp_loss * 2.0

                """ Loss term for whitening """
                # unet_loss = self.gw_criterion(outputs['whitened'], targets['whitened'])

                """ Total loss for GW detection """
                # Bits and bobs loss from the above set
                bits_and_bobs_loss = emphasis_loss + snr_loss + mchirp_loss + dchirp_loss

                # Weighted BCE loss (if necessary)
                mse_weight = {}
                if self.weighted_bce_loss:
                    bce_loss = 0.0
                    weighted_bce_data = targets['weighted_bce_data']
                    num_params = len(weighted_bce_data.keys())
                    for param, data in weighted_bce_data.items():
                        counts, bins = data
                        counts = np.insert(counts, 0, np.sum(counts), axis=0)
                        assert 0 not in counts, "INVALID: Found a zero count bin in weighted BCE loss."
                        bins = np.insert(bins, 0, -2, axis=0)
                        # Include noise to the weighting as it is one of the classes
                        # Weights are calculated based on the num of samples in input dataset
                        # Add the weights for each parameter specified within weighted bce loss
                        # For noise, we add the same weight for each param given
                        total_count = np.sum(counts)
                        # Calculate weights using counts as 1/nsamples_in_bin
                        # Weight for noise will be 1/(0.5*total_count)
                        # We can convert these weights into effective weights by using percent of samples
                        # Finally, divide total calculated loss by number of parameters (num_params)
                        ## We iterate through each sample and use required weight
                        bin_idx = np.digitize(targets[param], bins)
                        weight = 1./(counts[bin_idx]/total_count)
                        mse_weight['norm_'+param] = [foo for foo in weight if foo != 2.]
                        weighted_criterion = torch.nn.BCEWithLogitsLoss(weight=weight)
                        bce_loss += weighted_criterion(outputs['raw'], targets['gw'])
                else:
                    bce_loss = self.gw_criterion(outputs['raw'], targets['gw'])
                
                # Additional Focal Loss
                # focal_loss = torchvision.ops.sigmoid_focal_loss(outputs['raw'], targets['gw'], reduction='mean')

                # Total BCE Loss
                BCEgw = bce_loss + bits_and_bobs_loss

            elif isinstance(self.gw_criterion, torch.nn.BCELoss):
                if self.noise_emphasis or self.signal_emphasis or self.snr_loss or self.mchirp_loss:
                    raise NotImplementedError('Bits and bobs loss not implemented for BCELoss')
                BCEgw = self.gw_criterion(outputs['pred_prob'], targets['gw'])
        
        else:
            if self.noise_emphasis or self.signal_emphasis or self.snr_loss or self.mchirp_loss:
                raise NotImplementedError('Bits and bobs loss not implemented for regularised losses')
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
            if self.mse_alpha == 0.0:
                pe_loss = torch.tensor(0.0).to(device='cuda:{}'.format(BCEgw.get_device()))
                custom_loss[key] = pe_loss
                MSEpe += pe_loss
                continue

            mask = torch.ge(targets[key], 0.0)
            masked_target = torch.masked_select(targets[key], mask)
            masked_output = torch.masked_select(outputs[key], mask)
            assert -1 not in masked_target, 'Found invalid value (-1) in PE target. Noise sample may have leaked!'
            if len(masked_target) == 0:
                pe_loss = torch.tensor(0.0)
            # Calculating the individual PE MSE loss
            elif self.variance_loss:
                # Variance loss on PE terms
                assert key+"_var" in outputs.keys(), "Variance output not present for {} parameter!".format(key)
                # TODO: Output log(sigma) instead of sigma for HDR.
                param_var = torch.masked_select(outputs[key+'_var'], mask)
                pe_loss = torch.mean(0.5 * ((masked_target-masked_output)/torch.exp(param_var))**2 + param_var)
                pe_loss = self.mse_alpha * pe_loss
            else:
                pe_loss = self.mse_alpha * torch.mean((masked_target-masked_output)**2)

            # Store losses
            if torch.is_tensor(pe_loss) and torch.isnan(pe_loss):
                raise ValueError("PE Loss for {} is nan! val = {}".format(key, pe_loss))
            if not torch.is_tensor(pe_loss):
                if np.isnan(pe_loss):
                    raise ValueError("PE Loss for {} is nan! val = {}".format(key, pe_loss))
            
            custom_loss[key] = pe_loss
            MSEpe += pe_loss

        
        """ 
        CUSTOM LOSS FUNCTION
        L = BCE(P_0) + alpha * MSE(P_1)
        """
        custom_loss['total_loss'] = BCEgw + MSEpe
        
        return custom_loss


