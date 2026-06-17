#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : legacy.py
Description     : Short description of the file

Created on 2025-11-06 12:57:18

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import torch
from torch import nn

# LOCAL
from ..backend.resnet2d_cbam import (
    resnet18_cbam,
    resnet34_cbam,
    resnet50_cbam,
    resnet101_cbam,
    resnet152_cbam,
)

from ..frontend.mscnn1d_cbam import ConvBlock, _initialize_frontend_weights

from sage.core.config import get_cfg, get_data_cfg

import torch.nn.functional as F
from typing import NamedTuple
from .consistency import (
    PerDetHead,
    consistency_statistic,
    corroboration_features,
)
from sage.core.detectors import pairwise_light_travel_times


class MSCNN1D_2DResNetCBAM(nn.Module):
    """
    Multi-scale CNN backend + ResNet CBAM frontend for GW detection.

    Args:
        backend_filters: base filter size for ConvBlock backend
        backend_kernel: base kernel size for ConvBlock backend
        resnet_size: 18, 34, 50, 101, 152
        norm_type: 'batchnorm', 'layernorm', 'instancenorm'
        num_point_estimates: number of continuous parameters to predict
    """

    def __init__(
        self,
        frontend_filters: int = 32,
        frontend_kernel: int = 64,
        backend_resnet_size: int = 50,
        norm_type: str = "groupnorm",
        dropout: float = 0.0,
    ):
        super().__init__()

        # Shared configs
        cfg = get_cfg()

        self.num_detectors = len(cfg.detectors)

        # Normalization layer — normalises each detector channel independently.
        # Use self.num_detectors instead of hardcoding 2 so this works for any
        # detector network (e.g. H1+L1+V1 → 3 detectors).
        norm_layers = {
            "batchnorm": nn.BatchNorm1d(self.num_detectors),
            "layernorm": nn.LayerNorm(self.num_detectors),
            "instancenorm": nn.InstanceNorm1d(self.num_detectors, affine=True),
            # One group over all detector channels: normalises the detectors
            # jointly, preserving their RELATIVE scale (unlike instancenorm,
            # which unit-normalises each detector independently).
            "groupnorm": nn.GroupNorm(1, self.num_detectors),
        }
        self.norm = norm_layers[norm_type]

        # CNN Frontend per detector
        self.frontend = nn.ModuleList(
            [
                ConvBlock(frontend_filters, frontend_kernel, dropout=dropout)
                for _ in range(self.num_detectors)
            ]
        )

        # ResNet Backend
        resnet_factories = {
            18: resnet18_cbam,
            34: resnet34_cbam,
            50: resnet50_cbam,
            101: resnet101_cbam,
            152: resnet152_cbam,
        }

        if backend_resnet_size not in resnet_factories:
            raise ValueError("resnet_size must be one of 18, 34, 50, 101, 152")
        self.backend = resnet_factories[backend_resnet_size](
            pretrained=False, dropout=dropout
        )

        # Feature pooling
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(512)
        self.flatten = nn.Flatten(start_dim=1)

        # Output layers
        self.get_ranking_statistic = nn.Linear(512, 1)

        # Create a Linear layer for each point estimate
        num_point_estimates = len(cfg.do_point_estimate)
        self.point_estimate_layers = nn.ModuleList(
            [nn.Linear(512, 1) for _ in range(num_point_estimates)]
        )

        # Initialising weights
        self._initialise_weights()

    def _initialise_weights(self):
        nn.init.normal_(self.get_ranking_statistic.weight, 0, 0.01)
        nn.init.zeros_(self.get_ranking_statistic.bias)

        for layer in self.point_estimate_layers:
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.zeros_(layer.bias)

        for det in self.frontend:
            _initialize_frontend_weights(det)

    def forward(self, x):
        """
        x: Tensor of shape (batch, 2, signal_length)
        returns: raw, pred_prob, point_estimates (list of tensors)
        """
        # Normalize input
        x = self.norm(x)

        # CNN Frontend
        cnn_outputs = [
            detector(x[:, i : i + 1]) for i, detector in enumerate(self.frontend)
        ]
        cnn_output = torch.cat(cnn_outputs, dim=1)

        # 2D ResNet CBAM Backend
        features = self.backend(cnn_output)
        features = self.flatten(self.avg_pool_1d(features))

        # Outputs
        ranking_statistic = self.get_ranking_statistic(features)

        # Each point estimate has its own Linear layer
        point_estimates = torch.cat(
            [layer(features) for layer in self.point_estimate_layers],
            dim=1,
        )

        return ranking_statistic, point_estimates


class MSCNN1D_2DResNetCBAM_Heteroscedastic(nn.Module):
    """
    Multi-scale CNN frontend + ResNet CBAM backend for GW detection.
    Outputs:
        - Ranking statistic (BCE)
        - Point estimates (mean + log variance for heteroscedastic regression)
    """

    def __init__(
        self,
        frontend_filters: int = 32,
        frontend_kernel: int = 64,
        backend_resnet_size: int = 50,
        norm_type: str = "groupnorm",
        dropout: float = 0.0,
    ):
        super().__init__()

        cfg = get_cfg()
        self.num_detectors = len(cfg.detectors)

        # Normalization layer — normalises each detector channel independently.
        # Use self.num_detectors instead of hardcoding 2 so this works for any
        # detector network (e.g. H1+L1+V1 → 3 detectors).
        norm_layers = {
            "batchnorm": nn.BatchNorm1d(self.num_detectors),
            "layernorm": nn.LayerNorm(self.num_detectors),
            "instancenorm": nn.InstanceNorm1d(self.num_detectors, affine=True),
            # One group over all detector channels: normalises the detectors
            # jointly, preserving their RELATIVE scale (unlike instancenorm,
            # which unit-normalises each detector independently).
            "groupnorm": nn.GroupNorm(1, self.num_detectors),
        }
        self.norm = norm_layers[norm_type]

        # CNN Frontend per detector
        self.frontend = nn.ModuleList(
            [
                ConvBlock(frontend_filters, frontend_kernel, dropout=dropout)
                for _ in range(self.num_detectors)
            ]
        )

        # ResNet Backend
        resnet_factories = {
            18: resnet18_cbam,
            34: resnet34_cbam,
            50: resnet50_cbam,
            101: resnet101_cbam,
            152: resnet152_cbam,
        }
        if backend_resnet_size not in resnet_factories:
            raise ValueError("resnet_size must be one of 18, 34, 50, 101, 152")
        self.backend = resnet_factories[backend_resnet_size](
            pretrained=False, dropout=dropout
        )

        # Feature pooling
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(512)
        self.flatten = nn.Flatten(start_dim=1)

        # Output layers
        self.get_ranking_statistic = nn.Linear(512, 1)

        # Heteroscedastic point estimates: mean + log variance per PE
        num_point_estimates = len(cfg.do_point_estimate)
        self.point_estimate_layers = nn.ModuleList(
            [nn.Linear(512, 2) for _ in range(num_point_estimates)]  # 2 = mu + log_var
        )

        # Initialize weights
        self._initialise_weights()

    def _initialise_weights(self):
        nn.init.normal_(self.get_ranking_statistic.weight, 0, 0.01)
        nn.init.zeros_(self.get_ranking_statistic.bias)

        for layer in self.point_estimate_layers:
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.zeros_(layer.bias)

        for det in self.frontend:
            _initialize_frontend_weights(det)

    def forward(self, x):
        """
        x: Tensor of shape (B, num_detectors=2, signal_length)
        Returns:
            ranking_statistic: (B, 1)
            point_estimates: (B, 2*num_pe)
                Blocked format: [mu_0, mu_1, ..., sraw_0, sraw_1, ...]
                (all predicted means first, then all raw sigma params).
                The raw sigma params are mapped to a strictly-positive std via
                softplus inside BCEWithPEsigmaLoss (no exp(log_var) collapse).
                This matches the layout expected by BCEWithPEsigmaLoss and
                SageUncompiledValidation, which split at [:num_pe] / [num_pe:].
        """
        # Normalize input
        x = self.norm(x)

        # CNN Frontend per detector
        cnn_outputs = [
            detector(x[:, i : i + 1]) for i, detector in enumerate(self.frontend)
        ]
        cnn_output = torch.cat(cnn_outputs, dim=1)

        # 2D ResNet CBAM backend
        features = self.backend(cnn_output)
        features = self.flatten(self.avg_pool_1d(features))

        # Ranking statistic for BCE
        ranking_statistic = self.get_ranking_statistic(features)

        # Heteroscedastic PE predictions.
        # Each layer outputs (B, 2): [mu_k, sigma_raw_k].
        # We collect all mus first and all sigma params second so the concatenated
        # tensor has the blocked layout [mu_0, ..., mu_K, sraw_0, ..., sraw_K]
        # rather than the interleaved layout [mu_0, sraw_0, mu_1, sraw_1, ...].
        # BCEWithPEsigmaLoss splits at [:num_pe] for mu and [num_pe:] for sigma,
        # so interleaved would silently mix mu/sigma for num_pe > 1.
        raw = [layer(features) for layer in self.point_estimate_layers]
        mus = torch.cat([r[:, :1] for r in raw], dim=1)        # (B, num_pe)
        sigma_raw = torch.cat([r[:, 1:] for r in raw], dim=1)  # (B, num_pe)
        point_estimates = torch.cat([mus, sigma_raw], dim=1)   # (B, 2*num_pe)

        return ranking_statistic, point_estimates


class ConsistencyOutput(NamedTuple):
    """Forward output of :class:`MSCNN1D_2DResNetCBAM_Consistency`."""

    ranking_stat: torch.Tensor       # (B, 1)  classification logit
    point_estimates: torch.Tensor    # (B, 2*num_pe)  merged heteroscedastic PE
    mu_tc: torch.Tensor              # (B, D)  per-detector tc means (window-norm)
    sigma_tc: torch.Tensor           # (B, D)  per-detector tc std (>0)
    mu_mc: torch.Tensor              # (B, D)  per-detector mchirp means
    sigma_mc: torch.Tensor           # (B, D)  per-detector mchirp std (>0)
    s_tc: torch.Tensor               # (B,)    arrival-time consistency statistic
    s_mc: torch.Tensor               # (B,)    chirp-mass consistency statistic


class MSCNN1D_2DResNetCBAM_Consistency(nn.Module):
    """Heteroscedastic detection network + multi-detector consistency heads.

    Identical merged backbone (two frontends -> 2-D ResNet-CBAM -> ranking +
    heteroscedastic PE heads) to
    :class:`MSCNN1D_2DResNetCBAM_Heteroscedastic`, plus a shared
    :class:`~sage.architecture.network.consistency.PerDetHead` applied to *each*
    frontend output (pre-merge). The per-detector ``tc``/``mchirp`` estimates
    form an uncertainty-weighted consistency statistic ``(s_tc, s_mc)``; those,
    the per-detector sigmas and the attention entropies are concatenated into
    the ranking head input as a *learned* corroboration combiner (not a hard
    gate).

    Parameters
    ----------
    t_grid : torch.Tensor, shape ``(L,)``
        Physical time (seconds) of each multirate output sample, from
        :meth:`~sage.dsp.multirate_sampling.MultirateSampler.output_time_grid`.
        Adaptive-averaged to the frontend's time length to give the per-step
        ``t_position`` the ``tc`` soft-argmax sums over.
    frontend_filters, frontend_kernel, backend_resnet_size, norm_type
        As in the heteroscedastic model.
    head_hidden : int
        Hidden width of the per-detector head MLPs.
    """

    def __init__(
        self,
        t_grid: torch.Tensor,
        frontend_filters: int = 32,
        frontend_kernel: int = 64,
        backend_resnet_size: int = 50,
        norm_type: str = "groupnorm",
        head_hidden: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        cfg = get_cfg()
        self.num_detectors = len(cfg.detectors)
        if self.num_detectors < 2:
            raise ValueError("Consistency heads require at least 2 detectors")

        norm_layers = {
            "batchnorm": nn.BatchNorm1d(self.num_detectors),
            "layernorm": nn.LayerNorm(self.num_detectors),
            "instancenorm": nn.InstanceNorm1d(self.num_detectors, affine=True),
            # One group over all detector channels: normalises the detectors
            # jointly, preserving their RELATIVE scale (unlike instancenorm,
            # which unit-normalises each detector independently).
            "groupnorm": nn.GroupNorm(1, self.num_detectors),
        }
        self.norm = norm_layers[norm_type]

        self.frontend = nn.ModuleList(
            [
                ConvBlock(frontend_filters, frontend_kernel, dropout=dropout)
                for _ in range(self.num_detectors)
            ]
        )

        resnet_factories = {
            18: resnet18_cbam, 34: resnet34_cbam, 50: resnet50_cbam,
            101: resnet101_cbam, 152: resnet152_cbam,
        }
        if backend_resnet_size not in resnet_factories:
            raise ValueError("resnet_size must be one of 18, 34, 50, 101, 152")
        self.backend = resnet_factories[backend_resnet_size](
            pretrained=False, dropout=dropout
        )

        self.avg_pool_1d = nn.AdaptiveAvgPool1d(512)
        self.flatten = nn.Flatten(start_dim=1)

        # Infer the frontend output (C, T) from a dummy forward (no hardcoding).
        L = int(t_grid.shape[0])
        was_training = self.frontend[0].training
        self.frontend[0].eval()
        with torch.no_grad():
            feat = self.frontend[0](torch.zeros(2, 1, L))   # (2, 1, C, T)
        if was_training:
            self.frontend[0].train()
        feat_ch, feat_T = int(feat.shape[2]), int(feat.shape[3])

        # Shared per-detector head + the time of each of its T steps. tc is
        # window-NORMALISED (divided by the analysis-window length) so it lives
        # on the same ~unit scale as the standardised per-detector mchirp — a
        # single (sigma_min, sigma_max) then fits both, and the two NLL terms are
        # balanced. The soft-argmax sums over the normalised grid, so mu_tc and
        # the light-travel time are normalised consistently and the tc target
        # (normalised the same way in the training loop) matches.
        self.per_det_head = PerDetHead(feat_ch, hidden=head_hidden, dropout=dropout)
        self.tc_scale = float(get_data_cfg().sample_length_in_s)
        t_position = F.adaptive_avg_pool1d(
            t_grid.to(torch.float32).view(1, 1, -1), feat_T
        ).view(-1) / self.tc_scale
        self.register_buffer("t_position", t_position, persistent=False)

        # Light-travel time of the (first) detector pair, exact from geometry,
        # normalised by the same window length as tc.
        ltt = float(pairwise_light_travel_times(cfg.detectors)[0, 1]) / self.tc_scale
        self.register_buffer(
            "light_travel_time", torch.tensor(ltt, dtype=torch.float32),
            persistent=False,
        )

        # Ranking head consumes the backbone features (512, left RAW — they are
        # the primary signal) + 8 corroboration features. The corroboration block
        # is a coherence *refinement* and must stay subordinate to the backbone.
        # It is LayerNorm'd (NOT BatchNorm): the s statistics are heavy-tailed
        # (chi-square-like) and the number of incoherent high-s samples varies
        # per batch with p_non_astrophysical, so batch statistics jump around —
        # LayerNorm is per-sample and immune to that, and tames the per-sample
        # tail without clipping it (the high-s tail is the discriminating signal).
        # The s features are already log1p-compressed upstream; LayerNorm then
        # equalises the block, and a small affine gain (corr_gain_init) starts it
        # at/below the backbone scale (~0.27). The gain is learnable.
        self.n_corr = 8
        self.corr_gain_init = 0.25
        self.corr_norm = nn.LayerNorm(self.n_corr)
        self.get_ranking_statistic = nn.Linear(512 + self.n_corr, 1)

        num_point_estimates = len(cfg.do_point_estimate)
        self.point_estimate_layers = nn.ModuleList(
            [nn.Linear(512, 2) for _ in range(num_point_estimates)]
        )

        self._initialise_weights()

    def _initialise_weights(self):
        nn.init.normal_(self.get_ranking_statistic.weight, 0, 0.01)
        nn.init.zeros_(self.get_ranking_statistic.bias)
        for layer in self.point_estimate_layers:
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.zeros_(layer.bias)
        for det in self.frontend:
            _initialize_frontend_weights(det)
        # Start the corroboration block subordinate to the raw backbone: a small
        # LayerNorm gain so it enters the logit at/below the backbone scale.
        nn.init.constant_(self.corr_norm.weight, self.corr_gain_init)
        nn.init.zeros_(self.corr_norm.bias)

    def forward(self, x) -> ConsistencyOutput:
        x = self.norm(x)

        # Per-detector frontends; (B, 1, C, T) each.
        cnn_outputs = [
            detector(x[:, i : i + 1]) for i, detector in enumerate(self.frontend)
        ]

        # Per-detector heads on the (B, C, T) features (drop the singleton dim).
        per_det = [
            self.per_det_head(co.squeeze(1), self.t_position) for co in cnn_outputs
        ]

        # Uncertainty-weighted consistency statistic for the first detector pair.
        s_tc, s_mc = consistency_statistic(
            per_det[0], per_det[1], self.light_travel_time
        )
        corr = corroboration_features(per_det[0], per_det[1], s_tc, s_mc)  # (B, 8)

        # Merged backbone (unchanged).
        cnn_output = torch.cat(cnn_outputs, dim=1)          # (B, D, C, T)
        features = self.flatten(self.avg_pool_1d(self.backend(cnn_output)))  # (B, 512)

        # Ranking statistic from raw backbone features + the LayerNorm'd, gain-
        # scaled corroboration block (kept subordinate to the backbone). corr is
        # float32 (statistic stability); LayerNorm it there, then cast to the
        # backbone dtype (fp16 under autocast) for the concat.
        corr = self.corr_norm(corr).to(features.dtype)
        ranking_stat = self.get_ranking_statistic(
            torch.cat([features, corr], dim=1)
        )

        # Merged heteroscedastic PE (blocked [mu..., sigma_raw...]); the raw sigma
        # params become a softplus std inside BCEWithPEsigmaLoss (no exp collapse).
        raw = [layer(features) for layer in self.point_estimate_layers]
        mus = torch.cat([r[:, :1] for r in raw], dim=1)
        sigma_raw = torch.cat([r[:, 1:] for r in raw], dim=1)
        point_estimates = torch.cat([mus, sigma_raw], dim=1)

        # Stack per-detector outputs to (B, D) for the consistency loss.
        mu_tc = torch.stack([o.mu_tc for o in per_det], dim=1)
        sigma_tc = torch.stack([o.sigma_tc for o in per_det], dim=1)
        mu_mc = torch.stack([o.mu_mc for o in per_det], dim=1)
        sigma_mc = torch.stack([o.sigma_mc for o in per_det], dim=1)

        return ConsistencyOutput(
            ranking_stat=ranking_stat,
            point_estimates=point_estimates,
            mu_tc=mu_tc,
            sigma_tc=sigma_tc,
            mu_mc=mu_mc,
            sigma_mc=sigma_mc,
            s_tc=s_tc,
            s_mc=s_mc,
        )
