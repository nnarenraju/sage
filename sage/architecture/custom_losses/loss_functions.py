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


class BCEWithFARLoss(nn.Module):
    """
    FAR-targeted training loss combining four components:

    1. BCE (full batch)            — baseline classification gradient for all
                                     samples; preserves performance on easy /
                                     confident examples.
    2. Focal mix                   — within-batch amplification of hard
                                     examples (high-ranking BG or low-ranking
                                     signal) without abandoning easy ones.
                                     Blended as (1-focal_mix)*BCE + focal_mix*focal.
    3. Heteroscedastic regression  — identical to BCEWithPEsigmaLoss.
    4. pAUC loss                   — maintains a running circular buffer of
                                     background logits seen during training and
                                     directly maximises the fraction of signals
                                     that rank above the estimated FAR threshold.
                                     Acts as a differentiable proxy for
                                     sensitivity at the target FAR.
    5. Coupling loss               — identical to BCEWithPEsigmaLoss.

    The running buffer is registered as Module buffers so it is saved in
    checkpoints and moves to the correct device automatically.

    Parameters
    ----------
    regression_weight, coupling_weight : float
        Same as BCEWithPEsigmaLoss.
    focal_mix : float in [0, 1]
        Fraction of classification loss replaced by focal version.
        0 = pure BCE (no amplification), 1 = pure focal.
        Recommended starting point: 0.4.
    focal_gamma : float
        Focal loss gamma. Higher = stronger amplification of hard examples.
    pauc_weight : float
        Weight of the pAUC sensitivity term.
    far_buffer_size : int
        Number of background logit scalars retained in the circular buffer.
    target_far_quantile : float in (0, 1)
        Background distribution quantile used as the FAR threshold.
        0.9999 ≈ 1-in-10,000 background events; map this to a FAR by
        estimating the trigger rate in your training data.
    pauc_warmup : int
        Number of buffer entries required before the pAUC term activates.
    """

    def __init__(
        self,
        regression_weight:    float = 1.0,
        coupling_weight:      float = 1.0,
        focal_mix:            float = 0.4,
        focal_gamma:          float = 2.0,
        pauc_weight:          float = 0.1,
        far_buffer_size:      int   = 100_000,
        target_far_quantile:  float = 0.9999,
        pauc_warmup:          int   = 5_000,
        eps:                  float = 1e-6,
    ):
        super().__init__()

        cfg = get_cfg()
        self.regression_weight   = regression_weight
        self.coupling_weight     = coupling_weight
        self.focal_mix           = focal_mix
        self.focal_gamma         = focal_gamma
        self.pauc_weight         = pauc_weight
        self.target_far_quantile = target_far_quantile
        self.pauc_warmup         = pauc_warmup
        self.eps                 = eps
        # total, bce, reg, coupling, pauc, focal = 6 components
        self.num_components = len(cfg.do_point_estimate) + 4

        # Running circular buffer of background logit values (CPU, float32).
        # Kept on CPU to avoid wasting GPU memory on scalar accumulation.
        # We manage it manually (not register_buffer) so it stays CPU-side.
        self._bg_buf    = torch.full((far_buffer_size,), -1e9, dtype=torch.float32)
        self._bg_ptr    = 0
        self._bg_filled = 0

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def _update_bg_buffer(self, logits: torch.Tensor):
        """Append background logit values (detached) to the circular buffer."""
        vals = logits.detach().float().cpu()
        n    = vals.numel()
        cap  = self._bg_buf.numel()

        end = self._bg_ptr + n
        if end <= cap:
            self._bg_buf[self._bg_ptr:end] = vals
        else:
            # Wrap around
            tail = cap - self._bg_ptr
            self._bg_buf[self._bg_ptr:] = vals[:tail]
            self._bg_buf[:n - tail]      = vals[tail:]

        self._bg_ptr    = (self._bg_ptr + n) % cap
        self._bg_filled = min(self._bg_filled + n, cap)

    def _far_threshold(self) -> torch.Tensor:
        """Return the buffer quantile that estimates the FAR threshold."""
        valid = self._bg_buf[:self._bg_filled]
        return torch.quantile(valid, self.target_far_quantile)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, outputs, targets):
        """
        Compute the full 5-component FAR-targeted loss.

        Parameters
        ----------
        outputs : tuple
            ``(ranking_stat, point_estimates)``
            ``ranking_stat``: shape ``(B,)`` — raw classification logits.
            ``point_estimates``: shape ``(B, 2 * num_pe)`` — predicted
            means and log-variances (same layout as
            :class:`BCEWithPEsigmaLoss`).
        targets : torch.Tensor, shape ``(B, num_pe + 1)``
            Last column is the binary class label; preceding columns are
            the physical regression targets.

        Returns
        -------
        torch.Tensor, shape ``(6,)``
            ``[total_loss, bce_loss, reg_loss, coupling_loss,
            pauc_loss, focal_loss]``.

        Notes
        -----
        The pAUC term activates only after ``pauc_warmup`` background
        logits have been accumulated in the internal circular buffer.
        Before that point it contributes zero to the total loss.
        """
        ranking_stat, point_estimates = outputs

        class_target  = targets[:, -1].to(ranking_stat.dtype)
        ranking_stat  = ranking_stat.reshape(-1)

        bg_mask  = class_target < 0.5
        sig_mask = class_target > 0.5

        # ── 1. Standard BCE (all samples) ─────────────────────────────
        bce_loss = F.binary_cross_entropy_with_logits(ranking_stat, class_target)

        # ── 2. Focal amplification (hard examples) ────────────────────
        p = torch.sigmoid(ranking_stat)
        # per-sample focal weight: amplify hard BG (high p) and hard sig (low p)
        focal_w = torch.where(
            sig_mask,
            (1.0 - p) ** self.focal_gamma,   # missed detection weight
            p          ** self.focal_gamma,   # false alarm weight
        )
        focal_per_sample = focal_w * F.binary_cross_entropy_with_logits(
            ranking_stat, class_target, reduction="none"
        )
        focal_loss = focal_per_sample.mean()
        cls_loss   = (1.0 - self.focal_mix) * bce_loss + self.focal_mix * focal_loss

        # ── 3. Heteroscedastic regression ─────────────────────────────
        pe_targets  = targets[:, :-1]
        signal_mask = targets[:, -1].unsqueeze(1)
        num_pe      = pe_targets.shape[1]

        mu      = point_estimates[:, :num_pe]
        log_var = point_estimates[:, num_pe:]
        log_var = torch.clamp(log_var, -10.0, 6.0)
        var     = torch.exp(log_var) + self.eps

        nll           = 0.5 * (log_var + (pe_targets - mu) ** 2 / var)
        p_signal      = torch.sigmoid(ranking_stat).detach().unsqueeze(1) ** 2
        nll           = nll * signal_mask * p_signal
        variance_reg  = var * signal_mask * p_signal
        num_signal    = signal_mask.sum().clamp_min(1.0)

        nll_loss          = nll.sum() / num_signal
        variance_reg_loss = variance_reg.sum() / num_signal
        reg_loss          = nll_loss + 1e-3 * variance_reg_loss

        # ── 4. Coupling loss ──────────────────────────────────────────
        mean_sigma    = torch.sqrt(var.mean(dim=1))
        coupling_loss = (mean_sigma * torch.sigmoid(ranking_stat)).mean()

        # ── 5. pAUC loss (FAR-targeted sensitivity) ───────────────────
        # Feed background logits into the circular buffer every step.
        if bg_mask.any():
            self._update_bg_buffer(ranking_stat[bg_mask])

        pauc_loss = ranking_stat.new_tensor(0.0)
        if self._bg_filled >= self.pauc_warmup and sig_mask.any():
            threshold = self._far_threshold().to(ranking_stat.device)
            # Soft sensitivity: fraction of signals ranked above FAR threshold.
            # Gradient flows back through sigmoid(sig - threshold) w.r.t. sig.
            sig_logits = ranking_stat[sig_mask]
            pauc_loss  = -torch.sigmoid(sig_logits - threshold).mean()

        # ── Total ─────────────────────────────────────────────────────
        total_loss = (
            cls_loss
            + self.regression_weight * reg_loss
            + self.coupling_weight   * coupling_loss
            + self.pauc_weight       * pauc_loss
        )

        return torch.stack(
            [total_loss, bce_loss, reg_loss, coupling_loss, pauc_loss, focal_loss],
            dim=0,
        )
