#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
networks.py -- production model(s) with a hard-mining frontend-embedding tap.

``MSCNN1D_2DResNetCBAM_HardMining`` is the heteroscedastic GW-detection model
(multi-scale 1D CNN frontend per detector -> 2D ResNet-CBAM backend -> ranking
statistic + heteroscedastic point estimates), copied from
``MSCNN1D_2DResNetCBAM_Heteroscedastic`` and extended with ONE thing: it exposes
the per-detector **frontend** embedding for hard-noise mining.

Why the frontend (not the backend / ranking-stat feature): the miner's
quality-diversity descriptor must keep distinct glitch families separable.  The
frontend output is the pre-backend morphology; the 512-d backend feature that
feeds ``get_ranking_statistic`` is collapsed toward the binary detection
decision and is a poor diversity space.

Compile-safety: ``train_hard`` compiles with ``fullgraph=True``, which forbids
forward hooks (they graph-break).  So instead of a hook, the embedding is built
inside ``forward`` only when ``self.return_frontend_embedding`` is set -- the
flag is ``False`` during (compiled) training, so the branch is pruned from the
graph and costs nothing; the hard-mining callback flips it ``True`` on the eager
model for the mining pass and reads ``self.frontend_embedding``.
"""

import torch
from torch import nn
import torch.nn.functional as F

from ..backend.resnet2d_cbam import (
    resnet18_cbam,
    resnet34_cbam,
    resnet50_cbam,
    resnet101_cbam,
    resnet152_cbam,
)
from ..frontend.mscnn1d_cbam import ConvBlock, _initialize_frontend_weights

from sage.core.config import get_cfg


class MSCNN1D_2DResNetCBAM_HardMining(nn.Module):
    """Multi-scale CNN frontend + ResNet-CBAM backend for GW detection, with a
    frontend-embedding tap for hard-noise mining.

    Outputs (unchanged from the heteroscedastic model):
        - Ranking statistic (BCE)
        - Point estimates (mean + raw sigma per PE, blocked layout)

    Extra: when ``return_frontend_embedding`` is True, ``forward`` also populates
    ``self.frontend_embedding`` -- the per-detector frontend features, global-
    pooled over the feature map, L2-normalised per detector and flattened to
    ``(B, D*C)``.  The hard-mining miner uses this as its diversity descriptor.
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
        norm_layers = {
            "batchnorm": nn.BatchNorm1d(self.num_detectors),
            "layernorm": nn.LayerNorm(self.num_detectors),
            "instancenorm": nn.InstanceNorm1d(self.num_detectors, affine=True),
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
        # The frontend emits one channel per detector (concatenated below), so
        # the backend's input channels must match num_detectors — otherwise a
        # 3-detector (HLV) network feeds 3 channels into a conv that defaults to
        # in_channels=2 and crashes on the first forward.
        self.backend = resnet_factories[backend_resnet_size](
            pretrained=False, in_channels=self.num_detectors, dropout=dropout
        )

        # Feature pooling
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(512)
        self.flatten = nn.Flatten(start_dim=1)

        # Output layers
        self.get_ranking_statistic = nn.Linear(512, 1)

        # Heteroscedastic point estimates: mean + raw sigma per PE
        num_point_estimates = len(cfg.do_point_estimate)
        self.point_estimate_layers = nn.ModuleList(
            [nn.Linear(512, 2) for _ in range(num_point_estimates)]
        )

        # Hard-mining frontend-embedding tap (off by default -> zero training cost)
        self.return_frontend_embedding = False
        self.frontend_embedding = None

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

    # Per-detector frontend feature maps are single-channel 2D "images"
    # (B, 1, H, W); a flat global pool would collapse each to one number. Adaptive
    # -pool to a small GRID x GRID instead so the descriptor keeps the coarse
    # time-frequency morphology that separates glitch families.
    _DESCRIPTOR_GRID = 8

    @classmethod
    def _frontend_descriptor(cls, cnn_outputs):
        """Per-detector frontend feature maps -> ``(B, D*K)`` diversity descriptor.

        Adaptive-pool each detector's feature map to a fixed grid, flatten,
        L2-norm per detector, then stack and flatten.  ``K = C * grid**2`` (2D)
        or ``C * grid`` (1D).
        """
        g = cls._DESCRIPTOR_GRID
        pooled = []
        for o in cnn_outputs:
            o = o.float()
            if o.dim() == 4:                       # (B, C, H, W)
                o = F.adaptive_avg_pool2d(o, (g, g))
            elif o.dim() == 3:                     # (B, C, L)
                o = F.adaptive_avg_pool1d(o, g)
            o = o.flatten(1)                       # (B, K)
            pooled.append(F.normalize(o, dim=-1))  # unit per detector
        return torch.stack(pooled, dim=1).flatten(1)   # (B, D*K)

    def forward(self, x):
        """
        x: Tensor of shape (B, num_detectors, signal_length)
        Returns:
            ranking_statistic: (B, 1)
            point_estimates:   (B, 2*num_pe)  -- [mu_0..mu_K, sraw_0..sraw_K]
        """
        # Normalize input
        x = self.norm(x)

        # CNN Frontend per detector
        cnn_outputs = [
            detector(x[:, i : i + 1]) for i, detector in enumerate(self.frontend)
        ]
        cnn_output = torch.cat(cnn_outputs, dim=1)

        # Hard-mining diversity descriptor from the FRONTEND (flag pruned from the
        # compiled training graph; only computed on the eager mining pass).
        if self.return_frontend_embedding:
            self.frontend_embedding = self._frontend_descriptor(cnn_outputs)

        # 2D ResNet CBAM backend
        features = self.backend(cnn_output)
        features = self.flatten(self.avg_pool_1d(features))

        # Ranking statistic for BCE
        ranking_statistic = self.get_ranking_statistic(features)

        # Heteroscedastic PE predictions (blocked layout: all mus, then all sigmas)
        raw = [layer(features) for layer in self.point_estimate_layers]
        mus = torch.cat([r[:, :1] for r in raw], dim=1)        # (B, num_pe)
        sigma_raw = torch.cat([r[:, 1:] for r in raw], dim=1)  # (B, num_pe)
        point_estimates = torch.cat([mus, sigma_raw], dim=1)   # (B, 2*num_pe)

        return ranking_statistic, point_estimates
