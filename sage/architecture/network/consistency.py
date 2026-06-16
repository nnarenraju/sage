#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-detector consistency heads.

Per-detector parameter heads that hang off *each* frontend (pre-merge), used to
build an uncertainty-weighted coherence statistic between detectors. A real GW
is coherent — its arrival times differ by at most the inter-detector
light-travel time and its chirp mass is shared — so disagreement that is large
*relative to the predicted uncertainty* is evidence against a real coincidence.

Components
----------
- :class:`AttentionPool1d` : learned temporal soft-attention pooling. Returns
  the pooled feature, the attention weights, the raw scores, and the attention
  *entropy* (a confidence signal: low entropy = peaked = confident).
- :class:`GlobalAvgMaxPool1d` : concat of temporal mean and max.
- :class:`PerDetHead` : per-detector ``tc`` (soft-argmax over physical time,
  attention-entropy-driven sigma) and ``mchirp`` (attention + avg/max pooled,
  MLP mean and sigma) heads, with their attention entropies.

All tensors follow the ``(B, C, T)`` convention of the Sage frontend output
(channels then time). Everything here is ``torch.compile``-safe (no
data-dependent control flow; config flags are static Python booleans).
"""

from typing import NamedTuple

import torch
import torch.nn as nn


class AttentionPool1d(nn.Module):
    """Learned temporal soft-attention pooling over the time axis.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    eps : float
        Stabiliser inside the entropy logarithm.

    Forward
    -------
    x : torch.Tensor, shape ``(B, C, T)``

    Returns
    -------
    pooled : ``(B, C)``      attention-weighted feature
    attn   : ``(B, T)``      softmax attention over time
    scores : ``(B, T)``      raw (pre-softmax) scores
    entropy: ``(B,)``        attention entropy ``-sum attn*log(attn+eps)``
                              (low = peaked/confident, high = diffuse/uncertain)
    """

    def __init__(self, in_ch: int, eps: float = 1e-8):
        super().__init__()
        # Linear(in_ch -> 1) applied per time step == 1x1 conv over channels.
        self.score = nn.Conv1d(in_ch, 1, kernel_size=1)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        scores = self.score(x).squeeze(1)                       # (B, T)
        attn = torch.softmax(scores, dim=-1)                    # (B, T)
        pooled = torch.sum(x * attn.unsqueeze(1), dim=-1)       # (B, C)
        entropy = -torch.sum(attn * torch.log(attn + self.eps), dim=-1)  # (B,)
        return pooled, attn, scores, entropy


class GlobalAvgMaxPool1d(nn.Module):
    """Concatenate temporal mean and max pooling: ``(B, C, T) -> (B, 2C)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)


class PerDetOutput(NamedTuple):
    """Per-detector head outputs (all shape ``(B,)``)."""

    mu_tc: torch.Tensor
    log_sigma_tc: torch.Tensor
    mu_mc: torch.Tensor
    log_sigma_mc: torch.Tensor
    entropy_tc: torch.Tensor
    entropy_mc: torch.Tensor


class PerDetHead(nn.Module):
    """Per-detector ``tc`` and ``mchirp`` heads applied to one frontend output.

    The same module is applied to each detector's frontend output (weights may
    be shared across detectors; only the inputs differ). ``t_position`` is the
    physical (within-window) time in **seconds** of each of the ``T`` steps —
    the multirate/frontend downsampling means index != time, so the mapping must
    be supplied.

    ``tc`` head (localisation-preserving): a saliency conv produces a temporal
    softmax whose soft-argmax over ``t_position`` gives ``mu_tc`` in seconds.
    A separate attention pool drives ``log_sigma_tc`` from the pooled feature
    plus the attention entropy (diffuse attention -> larger sigma).

    ``mchirp`` head (time-tolerant): attention + global avg/max features feed an
    MLP for ``mu_mc``; a second MLP (plus attention entropy) gives
    ``log_sigma_mc``.

    Parameters
    ----------
    in_ch : int
        Channels of the frontend output.
    hidden : int
        Hidden width of the sigma/mean MLPs.
    log_sigma_clamp : tuple[float, float]
        Clamp range for both log-sigmas (stability).
    ensemble_tc : bool
        If True, ``mu_tc`` is the mean of the soft-argmax and the
        attention-weighted time estimate. Default False (soft-argmax only).
    eps : float
        Entropy logarithm stabiliser.
    """

    def __init__(
        self,
        in_ch: int,
        hidden: int = 128,
        log_sigma_clamp=(-7.0, 3.0),
        ensemble_tc: bool = False,
        eps: float = 1e-8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.eps = eps
        self.log_sigma_min, self.log_sigma_max = log_sigma_clamp
        self.ensemble_tc = bool(ensemble_tc)

        # --- tc head ---
        self.tc_saliency = nn.Conv1d(in_ch, 1, kernel_size=1)   # soft-argmax map
        self.tc_attn = AttentionPool1d(in_ch, eps)              # drives sigma
        self.tc_log_sigma = nn.Sequential(
            nn.Linear(in_ch + 1, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # --- mchirp head ---
        self.mc_attn = AttentionPool1d(in_ch, eps)
        self.mc_pool = GlobalAvgMaxPool1d()
        mc_feat_dim = in_ch + 2 * in_ch  # attn pooled (C) + avg/max (2C)
        self.mc_mu = nn.Sequential(
            nn.Linear(mc_feat_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.mc_log_sigma = nn.Sequential(
            nn.Linear(mc_feat_dim + 1, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, t_position: torch.Tensor) -> PerDetOutput:
        # x: (B, C, T);  t_position: (T,) physical seconds
        # --- tc: soft-argmax over physical time ---
        saliency = self.tc_saliency(x).squeeze(1)            # (B, T)
        w = torch.softmax(saliency, dim=-1)                  # (B, T)
        mu_tc = torch.sum(w * t_position.unsqueeze(0), dim=-1)  # (B,)

        attn_feat_tc, attn_tc, _, entropy_tc = self.tc_attn(x)
        if self.ensemble_tc:
            mu_tc_attn = torch.sum(attn_tc * t_position.unsqueeze(0), dim=-1)
            mu_tc = 0.5 * (mu_tc + mu_tc_attn)

        log_sigma_tc = self.tc_log_sigma(
            torch.cat([attn_feat_tc, entropy_tc.unsqueeze(1)], dim=-1)
        ).squeeze(1)
        log_sigma_tc = log_sigma_tc.clamp(self.log_sigma_min, self.log_sigma_max)

        # --- mchirp: time-tolerant attention + avg/max ---
        attn_feat_mc, _, _, entropy_mc = self.mc_attn(x)
        feat = torch.cat([attn_feat_mc, self.mc_pool(x)], dim=-1)  # (B, 3C)
        mu_mc = self.mc_mu(feat).squeeze(1)
        log_sigma_mc = self.mc_log_sigma(
            torch.cat([feat, entropy_mc.unsqueeze(1)], dim=-1)
        ).squeeze(1)
        log_sigma_mc = log_sigma_mc.clamp(self.log_sigma_min, self.log_sigma_max)

        return PerDetOutput(
            mu_tc=mu_tc,
            log_sigma_tc=log_sigma_tc,
            mu_mc=mu_mc,
            log_sigma_mc=log_sigma_mc,
            entropy_tc=entropy_tc,
            entropy_mc=entropy_mc,
        )


def consistency_statistic(
    out_a: PerDetOutput,
    out_b: PerDetOutput,
    light_travel_time: torch.Tensor,
    eps: float = 1e-8,
):
    """Uncertainty-weighted inter-detector consistency statistics for a pair.

    ``s_tc`` is the chi-square-like arrival-time disagreement *in excess of* the
    light-travel time, scaled by the combined per-detector variance — so it is
    tolerant of a faint (large-sigma) detector but strict when both are
    confident. ``s_mc`` is the analogous statistic for chirp mass (no
    light-travel allowance: a real signal shares mchirp exactly).

    Parameters
    ----------
    out_a, out_b : PerDetOutput
        Per-detector head outputs for detectors A and B.
    light_travel_time : torch.Tensor
        Scalar light-travel time (seconds) between the two detectors.
    eps : float
        Denominator stabiliser.

    Returns
    -------
    s_tc, s_mc : torch.Tensor, each ``(B,)``
    """
    var_tc = torch.exp(2.0 * out_a.log_sigma_tc) + torch.exp(2.0 * out_b.log_sigma_tc)
    var_mc = torch.exp(2.0 * out_a.log_sigma_mc) + torch.exp(2.0 * out_b.log_sigma_mc)
    d_tc = torch.relu((out_a.mu_tc - out_b.mu_tc).abs() - light_travel_time)
    s_tc = d_tc**2 / (var_tc + eps)
    s_mc = (out_a.mu_mc - out_b.mu_mc) ** 2 / (var_mc + eps)
    return s_tc, s_mc


def corroboration_features(
    out_a: PerDetOutput,
    out_b: PerDetOutput,
    s_tc: torch.Tensor,
    s_mc: torch.Tensor,
) -> torch.Tensor:
    """Build the ``(B, 8)`` corroboration feature block for the ranking head.

    ``[s_tc, s_mc, sigma_tc_A, sigma_tc_B, sigma_mc_A, sigma_mc_B,
       entropy_tc_A, entropy_tc_B]`` — the disagreement statistics together with
    the per-detector uncertainties and attention entropies, so the learned
    combiner can require small disagreement AND small sigma AND peaked (low
    entropy) attention on both detectors, rather than a hard gate on ``s``.
    """
    return torch.stack(
        [
            s_tc,
            s_mc,
            torch.exp(out_a.log_sigma_tc),
            torch.exp(out_b.log_sigma_tc),
            torch.exp(out_a.log_sigma_mc),
            torch.exp(out_b.log_sigma_mc),
            out_a.entropy_tc,
            out_b.entropy_tc,
        ],
        dim=1,
    )
