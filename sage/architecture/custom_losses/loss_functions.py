# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = custom_loss_functions.py
Description     = Repository of custom loss functions

Created on Fri Jan 28 19:08:44 2022

__author__      = nnarenraju
__copyright__   = Copyright 2021, Sage
__credits__     = nnarenraju
__license__     = GPL-3.0-or-later
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
        beta: float = 0.5,
        sigma_min: float = 1e-3,
        sigma_max: float = 10.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        cfg = get_cfg()  # grab config
        self.regression_weight = regression_weight
        self.coupling_weight = coupling_weight
        # beta-NLL exponent (Seitzer et al. 2022); softplus sigma bounds. These
        # NaN-proof the merged head (clamp before softplus)
        # the same way: no exp(log_var) blow-up, no overconfident sigma collapse.
        self.beta = float(beta)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.num_components = len(cfg.do_point_estimate) + 2
        self.eps = eps  # stability for variance

    def _sigma(self, raw):
        """Softplus-parameterised std: strictly
        positive, floored at ``sigma_min``, capped at ``sigma_max``. No ``exp`` so
        a momentarily large logit cannot collapse sigma toward zero and blow up
        the ``(mu - y)^2 / sigma^2`` term."""
        import torch.nn.functional as F

        return torch.clamp(
            F.softplus(raw) + self.sigma_min, self.sigma_min, self.sigma_max
        )

    def forward(self, outputs, targets):
        """
        Compute heteroscedastic BCE + NLL regression + coupling loss.

        Parameters
        ----------
        outputs : tuple
            ``(ranking_stat, point_estimates)``
            ``ranking_stat``: shape ``(B,)`` — raw classification logits.
            ``point_estimates``: shape ``(B, 2 * num_pe)`` — concatenation
            of predicted means ``μ`` (first ``num_pe`` columns) and raw
            ``σ`` parameters (last ``num_pe`` columns), the latter mapped to a
            strictly-positive std via softplus (:meth:`_sigma`).
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

        # float32 throughout: 1/sigma^2 can exceed fp16 range under autocast.
        mu = point_estimates[:, :num_pe].float()
        sigma = self._sigma(point_estimates[:, num_pe:].float())  # softplus std > 0
        y = pe_targets.float()
        var = sigma * sigma

        # Heteroscedastic Gaussian NLL (softplus sigma, no exp -> no collapse).
        nll = 0.5 * (mu - y) ** 2 / var + torch.log(sigma)

        # beta-NLL (Seitzer et al. 2022): scale each term by stop_grad(sigma^2beta).
        # Tempers the ~1/sigma^2 gradient that lets a briefly-overconfident head
        # explode, without biasing the optimum. This is the fix that stops pe_reg
        # diverging.
        if self.beta > 0.0:
            nll = nll * (var**self.beta).detach()

        # Softened confidence curriculum: learn PE where the net believes it is a
        # signal. First power (was squared) -> gentler gate.
        p_signal = torch.sigmoid(ranking_stat).float().detach().unsqueeze(1)

        # Apply masks; normalise by the signal count (curriculum re-weights within).
        nll = nll * signal_mask.float() * p_signal
        num_signal = signal_mask.float().sum().clamp_min(1.0)

        # No variance regulariser: the old `lambda_var * var` term pushed sigma
        # *down*, driving the overconfident collapse. softplus + beta-NLL make it
        # unnecessary (and harmful), so it is removed.
        reg_loss = nll.sum() / num_signal

        # ----------------------
        # Coupling loss
        # ----------------------
        mean_sigma = sigma.mean(dim=1)

        sigmoid_rank = torch.sigmoid(ranking_stat).float()

        coupling_loss = mean_sigma * sigmoid_rank
        coupling_loss = coupling_loss.mean()

        # ----------------------
        # Total loss
        # ----------------------
        # fp32 throughout (PE terms are fp32); cast bce so the stack is uniform.
        bce_loss = bce_loss.float()
        total_loss = (
            bce_loss
            + (self.regression_weight * reg_loss)
            + (self.coupling_weight * coupling_loss)
        )

        return torch.stack([total_loss, bce_loss, reg_loss, coupling_loss], dim=0)


