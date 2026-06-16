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
import torch.nn.functional as F


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
    """Per-detector head outputs (all shape ``(B,)``).

    ``sigma_*`` are the actual standard deviations (``> 0``), produced by a
    softplus parameterisation, not log-sigmas — softplus grows linearly so it
    cannot blow up the ``1/sigma^2`` precision the way ``exp(-2 log_sigma)`` can.
    """

    mu_tc: torch.Tensor
    sigma_tc: torch.Tensor
    mu_mc: torch.Tensor
    sigma_mc: torch.Tensor
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
    softmax whose soft-argmax over ``t_position`` gives ``mu_tc``. A separate
    attention pool drives ``sigma_tc`` from the pooled feature plus the attention
    entropy (diffuse attention -> larger sigma).

    ``mchirp`` head (time-tolerant): attention + global avg/max features feed an
    MLP for ``mu_mc``; a second MLP (plus attention entropy) gives ``sigma_mc``.

    Both sigmas use a softplus parameterisation (``sigma = softplus(raw) +
    sigma_min``, clamped to ``[sigma_min, sigma_max]``) rather than ``exp`` of a
    clamped log-sigma — softplus is linear in the tail, so it cannot drive the
    ``1/sigma^2`` precision to overflow.

    Parameters
    ----------
    in_ch : int
        Channels of the frontend output.
    hidden : int
        Hidden width of the sigma/mean MLPs.
    sigma_min, sigma_max : float
        Lower / upper bound on the predicted standard deviations (both sides
        bounded so the precision ``1/sigma^2`` and the variance ``sigma^2`` stay
        finite).
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
        sigma_min: float = 1e-3,
        sigma_max: float = 10.0,
        ensemble_tc: bool = False,
        eps: float = 1e-8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.eps = eps
        # sigma = softplus(raw) + sigma_min, clamped to [sigma_min, sigma_max].
        # Bounded on BOTH sides: the floor keeps 1/sigma^2 finite (no downward
        # explosion), the ceiling keeps sigma^2 finite (no upward overflow).
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
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

    def _sigma(self, raw: torch.Tensor) -> torch.Tensor:
        """Map a raw head output to a positive, bounded standard deviation.

        ``softplus`` grows *linearly* (unlike ``exp``), so a large raw value
        cannot send ``1/sigma^2`` to infinity; the ``+sigma_min`` and clamp bound
        it on both sides.
        """
        return (F.softplus(raw) + self.sigma_min).clamp(self.sigma_min, self.sigma_max)

    def forward(self, x: torch.Tensor, t_position: torch.Tensor) -> PerDetOutput:
        # x: (B, C, T);  t_position: (T,) physical (or window-normalised) time
        # --- tc: soft-argmax over physical time ---
        saliency = self.tc_saliency(x).squeeze(1)            # (B, T)
        w = torch.softmax(saliency, dim=-1)                  # (B, T)
        mu_tc = torch.sum(w * t_position.unsqueeze(0), dim=-1)  # (B,)

        attn_feat_tc, attn_tc, _, entropy_tc = self.tc_attn(x)
        if self.ensemble_tc:
            mu_tc_attn = torch.sum(attn_tc * t_position.unsqueeze(0), dim=-1)
            mu_tc = 0.5 * (mu_tc + mu_tc_attn)

        raw_sigma_tc = self.tc_log_sigma(
            torch.cat([attn_feat_tc, entropy_tc.unsqueeze(1)], dim=-1)
        ).squeeze(1)
        sigma_tc = self._sigma(raw_sigma_tc)

        # --- mchirp: time-tolerant attention + avg/max ---
        attn_feat_mc, _, _, entropy_mc = self.mc_attn(x)
        feat = torch.cat([attn_feat_mc, self.mc_pool(x)], dim=-1)  # (B, 3C)
        mu_mc = self.mc_mu(feat).squeeze(1)
        raw_sigma_mc = self.mc_log_sigma(
            torch.cat([feat, entropy_mc.unsqueeze(1)], dim=-1)
        ).squeeze(1)
        sigma_mc = self._sigma(raw_sigma_mc)

        return PerDetOutput(
            mu_tc=mu_tc,
            sigma_tc=sigma_tc,
            mu_mc=mu_mc,
            sigma_mc=sigma_mc,
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
    s_tc, s_mc : torch.Tensor, each ``(B,)`` (always float32)
    """
    # Force float32: d^2 / var can overflow fp16 under autocast. sigma is already
    # bounded to [sigma_min, sigma_max] by the head, so var is bounded on both
    # sides (no underflow to 0, no overflow to inf). Tiny (B,) tensors -> the fp32
    # cost is negligible.
    s_a_tc, s_b_tc = out_a.sigma_tc.float(), out_b.sigma_tc.float()
    s_a_mc, s_b_mc = out_a.sigma_mc.float(), out_b.sigma_mc.float()
    ltt = light_travel_time.float() if torch.is_tensor(light_travel_time) else light_travel_time

    var_tc = s_a_tc**2 + s_b_tc**2
    var_mc = s_a_mc**2 + s_b_mc**2
    # Smooth |.|: sqrt(d^2 + eps) instead of abs(d). abs has an undefined gradient
    # at 0 and bare sqrt(0) has an infinite one — both NaN gradient sources if the
    # two mu ever coincide. The eps inside the sqrt keeps the gradient finite.
    d_tc_abs = torch.sqrt((out_a.mu_tc.float() - out_b.mu_tc.float()) ** 2 + eps)
    d_mc_abs = torch.sqrt((out_a.mu_mc.float() - out_b.mu_mc.float()) ** 2 + eps)
    d_tc = torch.relu(d_tc_abs - ltt)
    s_tc = d_tc**2 / (var_tc + eps)
    s_mc = d_mc_abs**2 / (var_mc + eps)
    return s_tc, s_mc


def corroboration_features(
    out_a: PerDetOutput,
    out_b: PerDetOutput,
    s_tc: torch.Tensor,
    s_mc: torch.Tensor,
) -> torch.Tensor:
    """Build the ``(B, 8)`` corroboration feature block for the ranking head.

    ``[log1p(s_tc), log1p(s_mc), sigma_tc_A, sigma_tc_B, sigma_mc_A, sigma_mc_B,
       entropy_tc_A, entropy_tc_B]`` — the disagreement statistics together with
    the per-detector uncertainties and attention entropies, so the learned
    combiner can require small disagreement AND small sigma AND peaked (low
    entropy) attention on both detectors, rather than a hard gate on ``s``.

    The chi-square statistics are ``log1p``-compressed: ``s`` is an unbounded
    squared z-score (it can reach ~1e7 for a confident disagreement), and feeding
    it raw would overflow the fp16 ranking head and let a single feature dominate
    the classification logits. ``log1p`` is monotonic (large ``s`` still reads as
    "inconsistent") and bounded, and a log scale is friendlier to a linear head.
    Returned in float32; the caller casts to the ranking-head dtype.
    """
    return torch.stack(
        [
            torch.log1p(s_tc),
            torch.log1p(s_mc),
            out_a.sigma_tc.float(),
            out_b.sigma_tc.float(),
            out_a.sigma_mc.float(),
            out_b.sigma_mc.float(),
            out_a.entropy_tc.float(),
            out_b.entropy_tc.float(),
        ],
        dim=1,
    )
