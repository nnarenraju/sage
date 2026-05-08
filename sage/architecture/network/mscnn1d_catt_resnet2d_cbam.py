#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : mscnn1d_catt_resnet2d_cbam.py
Description     : Short description of the file

Created on 2026-03-19 23:50:10

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
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
import torch.nn as nn

import warnings

# LOCAL
from ..backend.resnet2d_cbam import (
    resnet18_cbam,
    resnet34_cbam,
    resnet50_cbam,
    resnet101_cbam,
    resnet152_cbam,
)

from ..frontend.mscnn1d_cbam import ConvBlock, _initialize_frontend_weights
from ..zoo.cross_attention import CrossAttention2D, AxialCrossAttention2D

from sage.core.config import get_cfg, get_data_cfg


class MSCNN1D_catt_2DResNetCBAM(nn.Module):
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
        cross_attention_heads: int = 4,
        backend_resnet_size: int = 50,
        norm_type: str = "instancenorm",
    ):
        super().__init__()

        # NOTE: This is strictly a 2 ifo architecture
        warnings.warn("MSCNN1D_catt_2DResNetCBAM is currently a 2 ifo architecture")

        # Shared configs
        cfg = get_cfg()

        self.num_detectors = len(cfg.detectors)

        # Normalization layer
        norm_layers = {
            "batchnorm": nn.BatchNorm1d(2),
            "layernorm": nn.LayerNorm(2),
            "instancenorm": nn.InstanceNorm1d(2, affine=True),
        }
        self.norm = norm_layers[norm_type]

        # CNN Frontend per detector
        self.frontend = nn.ModuleList(
            [
                ConvBlock(frontend_filters, frontend_kernel)
                for _ in range(self.num_detectors)
            ]
        )

        # Cross attention
        self.cross_attention = AxialCrossAttention2D(
            128,
            cross_attention_heads,
        )

        C = 128
        self.gate = nn.Sequential(
            nn.Conv2d(2 * C, C, 1),
            nn.ReLU(),
            nn.Conv2d(C, C, 1),
            nn.Sigmoid(),
        )

        self.proj_gate = nn.Conv2d(1, 128, kernel_size=1)
        self.reduce = nn.Conv2d(128, 1, kernel_size=1)

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
            pretrained=False, in_channels=3
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

        # Cross-attention (alignment-aware)
        # NOTE: This is built for 2 ifo framework
        f1_attn, f2_attn = self.cross_attention(*cnn_outputs)

        # Residual fusion
        # Project f1 and f2 to the feature dimension C (same as attention output)
        proj_f1 = self.proj_gate(cnn_outputs[0])  # (B, C, H, W)
        proj_f2 = self.proj_gate(cnn_outputs[1])  # (B, C, H, W)

        # Compute gate
        g = self.gate(torch.cat([proj_f1, proj_f2], dim=1))  # (B, 2*C, H, W)
        # Applying relational gating to attention
        # Residual is used for fallback in case attention does not work
        f1 = self.reduce(cnn_outputs[0] + g * f1_attn)
        f2 = self.reduce(cnn_outputs[1] + g * f2_attn)

        # relational features
        relational_features = torch.cat([f1, f2, torch.abs(f1 - f2)], dim=1)

        # 2D ResNet CBAM Backend
        features = self.backend(relational_features)
        features = self.flatten(self.avg_pool_1d(features))

        # Outputs
        ranking_statistic = self.get_ranking_statistic(features)

        # Each point estimate has its own Linear layer
        point_estimates = torch.cat(
            [layer(features) for layer in self.point_estimate_layers],
            dim=1,
        )

        return ranking_statistic, point_estimates
