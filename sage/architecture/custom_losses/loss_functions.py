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

# Packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# LOCAL
from sage.core.config import get_cfg


class BCEWithPEregLoss(nn.Module):

    def __init__(self, regression_weight: float = 1.0):
        super().__init__()

        # Weight between classification and regression
        self.regression_weight = regression_weight

        # Required for training tracker
        cfg = get_cfg()
        self.num_components = len(cfg.do_point_estimate) + 1

    def forward(self, outputs, targets):

        ranking_stat, point_estimates = outputs

        # Classification target
        class_target = targets[:, -1].to(ranking_stat.dtype)

        # BCE expects same shape
        ranking_stat = ranking_stat.reshape(-1)

        bce_loss = F.binary_cross_entropy_with_logits(
            ranking_stat,
            class_target,
        )

        # Regression targets
        pe_targets = targets[:, :-1]
        signal_mask = targets[:, -1].unsqueeze(1)

        # MSE loss with mean performed only for signal batch
        reg = F.smooth_l1_loss(point_estimates, pe_targets, reduction="none")
        reg = reg * signal_mask

        # This weights PE based on perceived signal probability
        p_signal = torch.sigmoid(ranking_stat).detach()
        reg = reg * p_signal.unsqueeze(1)
        reg_loss = reg.sum() / signal_mask.sum().clamp_min(1)

        # Final total loss BCE + weighted MSE regularisation
        total_loss = bce_loss + self.regression_weight * reg_loss

        return total_loss
