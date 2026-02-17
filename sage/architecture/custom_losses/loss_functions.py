# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = custom_loss_functions.py
Description     = Repository of custom loss functions

Created on Fri Jan 28 19:08:44 2022

__author__      = nnarenraju
__copyright__   = Copyright 2021, Sage
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


class BCEWithPEregLoss:

    def __init__(self, gw_loss=None, mse_alpha=1.0):

        super().__init__()
        # MSE Loss is always ON with PE
        assert mse_alpha >= 0.0, "mse_alpha must be greater than or equal to 0.0"
        # Set generic params
        self.mse_alpha = mse_alpha
        self.gw_loss = gw_loss

    def __str__(self):
        # Display details of loss function
        str = "Loss function = {}".format(self.gw_loss.__class__.__name__)
        return str

    def forward(self, outputs, targets, source_params, cfg):
        # BCE to check whether the signal contains GW or is pure noise
        # MSE for calculation of correct 'tc'
        custom_loss = {}
        BCEgw = self.gw_loss(outputs["raw"], targets["gw"])
        custom_loss["gw"] = BCEgw

        """
        MSE - Mean Squared Error Loss
        For the handling of 'tc'
        MSEloss = (alpha / N_batch) * SUMMATION (target_tc - pred_tc)^2 / variance_tc
        """
        MSEpe = 0
        if "parameter_estimation" in cfg.model_params.keys():
            if len(cfg.model_params["parameter_estimation"]) != 0:
                for key in cfg.model_params["parameter_estimation"]:
                    # Get a masked loss calculation for parameter estimation
                    # Ignore all targets corresponding to pure noise samples
                    if self.mse_alpha == 0.0:
                        pe_loss = torch.tensor(0.0).to(
                            device="cuda:{}".format(BCEgw.get_device())
                        )
                        custom_loss[key] = pe_loss
                        MSEpe += pe_loss
                        continue

                    mask = torch.ge(targets[key], 0.0)
                    masked_target = torch.masked_select(targets[key], mask)
                    masked_output = torch.masked_select(outputs[key], mask)
                    assert (
                        -1 not in masked_target
                    ), "Found invalid value (-1) in PE target. Noise sample may have leaked into signals!"
                    if len(masked_target) == 0:
                        pe_loss = torch.tensor(0.0)
                    else:
                        pe_loss = self.mse_alpha * torch.mean(
                            (masked_target - masked_output) ** 2
                        )

                    # Store losses
                    if torch.is_tensor(pe_loss) and torch.isnan(pe_loss):
                        raise ValueError(
                            "PE Loss for {} is nan! val = {}".format(key, pe_loss)
                        )
                    if not torch.is_tensor(pe_loss):
                        if np.isnan(pe_loss):
                            raise ValueError(
                                "PE Loss for {} is nan! val = {}".format(key, pe_loss)
                            )

                    custom_loss[key] = pe_loss
                    MSEpe += pe_loss

        """ 
        CUSTOM LOSS FUNCTION
        L = BCE(P_0) + alpha * MSE(P_1)
        """
        custom_loss["total_loss"] = BCEgw + MSEpe

        return custom_loss
