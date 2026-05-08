#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : mscnn1d_att_resnet3d_cbam.py
Description     : Short description of the file

Created on 2026-03-10 02:05:58

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
from torch import nn

# LOCAL
from ..backend.resnet3d_cbam import (
    resnet18_cbam,
    resnet34_cbam,
    resnet50_cbam,
    resnet101_cbam,
    resnet152_cbam,
)

from ..frontend.mscnn1d_cbam import ConvBlock, _initialize_frontend_weights

from sage.core.config import get_cfg, get_data_cfg


class MSCNN1Datt_3DResNetCBAM(nn.Module):
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
        norm_type: str = "instancenorm",
    ):
        super().__init__()

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
        self.backend = resnet_factories[backend_resnet_size](pretrained=False)

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
        cnn_output = torch.stack(cnn_outputs, dim=2)

        # 3D ResNet CBAM Backend
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
