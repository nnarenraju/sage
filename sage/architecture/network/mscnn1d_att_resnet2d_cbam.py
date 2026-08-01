#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : legacy.py
Description     : Short description of the file

Created on 2025-11-06 12:57:18

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
__license__       = GPL-3.0-or-later
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
        # BlurPool anti-aliasing and ResNet-C/D are toggleable via config (both
        # default ON). use_blurpool/use_resnet_cd = False reproduces the
        # pre-df55e89 (o3b_dummy_1) architecture exactly.
        use_blurpool = getattr(cfg, "use_blurpool", True)
        use_resnet_cd = getattr(cfg, "use_resnet_cd", True)

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
                ConvBlock(frontend_filters, frontend_kernel, dropout=dropout,
                          use_blurpool=use_blurpool)
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
            pretrained=False, dropout=dropout,
            use_blurpool=use_blurpool, use_resnet_cd=use_resnet_cd,
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
        # BlurPool anti-aliasing and ResNet-C/D are toggleable via config (both
        # default ON). use_blurpool/use_resnet_cd = False reproduces the
        # pre-df55e89 (o3b_dummy_1) architecture exactly.
        use_blurpool = getattr(cfg, "use_blurpool", True)
        use_resnet_cd = getattr(cfg, "use_resnet_cd", True)

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
                ConvBlock(frontend_filters, frontend_kernel, dropout=dropout,
                          use_blurpool=use_blurpool)
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
            pretrained=False, dropout=dropout,
            use_blurpool=use_blurpool, use_resnet_cd=use_resnet_cd,
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


