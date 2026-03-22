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

        return torch.stack([total_loss, bce_loss, reg_loss], dim=0)


class BCEWithPEsigmaLoss(nn.Module):
    """
    Combined BCE + Heteroscedastic Regression Loss.

    - BCE for classification (ranking statistic).
    - Regression term uses predicted mean and log-variance.
    - Only computed for signal entries.
    - Weighted per-sample by network's predicted signal probability.
    """

    def __init__(self, regression_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        cfg = get_cfg()  # grab config
        self.regression_weight = regression_weight
        self.num_components = len(cfg.do_point_estimate) + 1
        self.eps = eps  # stability for variance

    def forward(self, outputs, targets):
        """
        Parameters
        ----------
        outputs : tuple
            (ranking_stat, point_estimates)
            point_estimates.shape = (B, 2*num_pe) -> mean, log_var concatenated
        targets : torch.Tensor
            Shape (B, num_pe + 1) -> last column is class (0 or 1)
        """

        ranking_stat, point_estimates = outputs
        class_target = targets[:, -1].to(ranking_stat.dtype)
        ranking_stat = ranking_stat.reshape(-1)

        # ----------------------
        # Classification loss
        # ----------------------
        bce_loss = F.binary_cross_entropy_with_logits(
            ranking_stat,
            class_target,
        )

        # ----------------------
        # Regression loss (heteroscedastic)
        # ----------------------
        pe_targets = targets[:, :-1]  # (B, num_pe)
        signal_mask = targets[:, -1].unsqueeze(1)  # (B, 1)

        num_pe = pe_targets.shape[1]
        mu = point_estimates[:, :num_pe]  # predicted mean
        log_var = point_estimates[:, num_pe:]  # predicted log-variance
        var = torch.exp(log_var) + self.eps  # variance > 0

        # Mask for signal samples only
        mu_sig = mu * signal_mask
        var_sig = var * signal_mask
        pe_targets_sig = pe_targets * signal_mask

        # Negative log-likelihood per parameter
        reg = 0.5 * (torch.log(var_sig) + (pe_targets_sig - mu_sig) ** 2 / var_sig)

        # Weight regression by network's predicted signal probability
        p_signal = torch.sigmoid(ranking_stat).unsqueeze(1).detach()
        reg = reg * p_signal

        # Average over signal samples
        reg_loss = reg.sum() / signal_mask.sum().clamp_min(1)

        # ----------------------
        # Total loss
        # ----------------------
        total_loss = bce_loss + self.regression_weight * reg_loss

        # Return stacked for compatibility with logger/tracker
        return torch.stack([total_loss, bce_loss, reg_loss], dim=0)
