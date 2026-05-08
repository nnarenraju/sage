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
    """
    Binary cross-entropy classification loss with MSE-based parameter
    estimation regularisation.

    The total loss is::

        L = BCE(ranking_stat, class_target)
          + regression_weight * MSE_signal(point_estimates, pe_targets)

    where the MSE term is:

    * computed only on signal samples (``class_target == 1``),
    * weighted per-sample by the network's current predicted signal
      probability ``p = sigmoid(ranking_stat)`` to focus regression
      updates on confident detections.

    This is the simplest multi-task loss in Sage and does not model
    prediction uncertainty.

    Parameters
    ----------
    regression_weight : float
        Relative weight of the regression term vs. BCE.

    Returns
    -------
    torch.Tensor, shape ``(num_pe + 1,)``
        Stacked ``[total_loss, bce_loss, reg_loss, ...]`` (one entry per
        point-estimate parameter plus the total).
    """

    def __init__(self, regression_weight: float = 1.0):
        super().__init__()

        # Weight between classification and regression
        self.regression_weight = regression_weight

        # Required for training tracker
        cfg = get_cfg()
        self.num_components = len(cfg.do_point_estimate) + 1

    def forward(self, outputs, targets):
        """
        Compute BCE + MSE regression loss.

        Parameters
        ----------
        outputs : tuple
            ``(ranking_stat, point_estimates)``
            ``ranking_stat``: shape ``(B,)`` or ``(B, 1)`` — raw logits.
            ``point_estimates``: shape ``(B, num_pe)`` — predicted parameters.
        targets : torch.Tensor, shape ``(B, num_pe + 1)``
            Last column is the binary class label (0 = noise, 1 = signal).
            Preceding columns are the regression targets.

        Returns
        -------
        torch.Tensor, shape ``(num_pe + 1,)``
            ``[total_loss, bce_loss, reg_loss]``.
        """

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

    def __init__(
        self,
        regression_weight: float = 1.0,
        coupling_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        cfg = get_cfg()  # grab config
        self.regression_weight = regression_weight
        self.coupling_weight = coupling_weight
        self.num_components = len(cfg.do_point_estimate) + 2
        self.eps = eps  # stability for variance

    def forward(self, outputs, targets):
        """
        Compute heteroscedastic BCE + NLL regression + coupling loss.

        Parameters
        ----------
        outputs : tuple
            ``(ranking_stat, point_estimates)``
            ``ranking_stat``: shape ``(B,)`` — raw classification logits.
            ``point_estimates``: shape ``(B, 2 * num_pe)`` — concatenation
            of predicted means ``μ`` (first ``num_pe`` columns) and
            predicted log-variances ``log σ²`` (last ``num_pe`` columns).
        targets : torch.Tensor, shape ``(B, num_pe + 1)``
            Last column is the binary class label; preceding columns are
            the physical regression targets.

        Returns
        -------
        torch.Tensor, shape ``(num_pe + 2,)``
            ``[total_loss, bce_loss, reg_loss, coupling_loss]``.
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
        # Regression loss (heteroscedastic + variance regularisation)
        # ----------------------
        pe_targets = targets[:, :-1]
        signal_mask = targets[:, -1].unsqueeze(1)

        num_pe = pe_targets.shape[1]

        mu = point_estimates[:, :num_pe]
        log_var = point_estimates[:, num_pe:]

        # Bound uncertainty head
        log_var = torch.clamp(log_var, -10.0, 6.0)

        var = torch.exp(log_var) + self.eps

        # Gaussian NLL
        nll = 0.5 * (log_var + (pe_targets - mu) ** 2 / var)

        # Confidence curriculum
        p_signal = torch.sigmoid(ranking_stat).detach().unsqueeze(1)
        p_signal = p_signal**2

        # Apply masks
        nll = nll * signal_mask * p_signal

        # Variance regularisation (prevents sigma explosion)
        variance_reg = var * signal_mask * p_signal

        num_signal = signal_mask.sum().clamp_min(1.0)

        nll_loss = nll.sum() / num_signal
        variance_reg_loss = variance_reg.sum() / num_signal

        # Strength of variance regulariser
        lambda_var = 1e-3

        reg_loss = nll_loss + lambda_var * variance_reg_loss

        # ----------------------
        # Coupling loss
        # ----------------------
        mean_sigma = torch.sqrt(var.mean(dim=1))

        sigmoid_rank = torch.sigmoid(ranking_stat)

        coupling_loss = mean_sigma * sigmoid_rank
        coupling_loss = coupling_loss.mean()

        # ----------------------
        # Total loss
        # ----------------------
        total_loss = (
            bce_loss
            + (self.regression_weight * reg_loss)
            + (self.coupling_weight * coupling_loss)
        )

        return torch.stack([total_loss, bce_loss, reg_loss, coupling_loss], dim=0)


