#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : resnet1d_ppe.py
Description     : Short description of the file

Created on 2025-11-06 13:00:38

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

# PACKAGES
import timm
import json
import torch
from torch import nn
from datetime import date

# Importing architecture snippets from zoo
from sage.architecture.backend.resnet_cbam import (
    resnet50_cbam,
    resnet152_cbam,
    resnet34_cbam,
)
from sage.architecture.backend.res2net_v1b import (
    res2net101_v1b_26w_4s,
    res2net50_v1b_26w_4s,
    res2net152_v1b_26w_4s,
)
from sage.architecture.frontend.osnet1d import osnet_ain_custom as osnet1d
from sage.architecture.frontend.kaggle import ConvBlock, _initialize_weights
from sage.architecture.frontend.mscnn1d import MSFeatureExtractor, MultiScaleBlock
from sage.architecture.frontend.resnet1d import resnet50, resnet101, resnet152

# Code Review
from sage.utils.decorators import unreviewed_model
from sage.utils.review import set_review_date

# Datatype for storage
data_type = torch.float32


@unreviewed_model
class KappaModel_ResNet1D(torch.nn.Module):
    """
    Kappa-type Model PE Architecture with ResNet1D

    Description - consists of a ResNet1D backend

    """

    def __init__(
        self,
        model_name="resnet1d",
        resnet_size: int = 50,
        norm_layer: str = "instancenorm",
        parameter_estimation=(
            "norm_tc",
            "norm_mchirp",
        ),
        store_device: str = "cpu",
        **kwargs
    ):

        super().__init__()

        self.model_name = model_name
        self.store_device = store_device
        self.norm_layer = norm_layer

        """ 1D ResNet """
        if resnet_size == 50:
            self.resnet = resnet50(num_classes=512)
        elif resnet_size == 101:
            self.resnet = resnet101(num_classes=512)
        elif resnet_size == 152:
            self.resnet = resnet152(num_classes=512)

        """ Mods """
        # Manipulation layers
        self.batchnorm = nn.BatchNorm1d(2)
        self.instancenorm = nn.InstanceNorm1d(2, affine=True)
        self.flatten_d1 = nn.Flatten(start_dim=1)
        self.flatten_d0 = nn.Flatten(start_dim=0)
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(512)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

        ## Convert network into given dtype and store in proper device
        # Primary outputs
        self.signal_or_noise = nn.Linear(512, 1)
        self.coalescence_time = nn.Linear(512, 1)
        self.chirp_distance = nn.Linear(512, 1)
        self.chirp_mass = nn.Linear(512, 1)
        self.distance = nn.Linear(512, 1)
        self.mass_ratio = nn.Linear(512, 1)
        self.inv_mass_ratio = nn.Linear(512, 1)
        self.snr = nn.Linear(512, 1)
        # Mod layers
        self.signal_or_noise.to(dtype=data_type, device=self.store_device)
        self.coalescence_time.to(dtype=data_type, device=self.store_device)
        self.chirp_distance.to(dtype=data_type, device=self.store_device)
        self.chirp_mass.to(dtype=data_type, device=self.store_device)
        self.distance.to(dtype=data_type, device=self.store_device)
        self.mass_ratio.to(dtype=data_type, device=self.store_device)
        self.inv_mass_ratio.to(dtype=data_type, device=self.store_device)
        self.snr.to(dtype=data_type, device=self.store_device)

        # Manipulation layers
        self.batchnorm.to(dtype=data_type, device=self.store_device)
        self.instancenorm.to(dtype=data_type, device=self.store_device)
        # Main layers
        self.resnet.to(dtype=data_type, device=self.store_device)

    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        if self.norm_layer == "batchnorm":
            normed = self.batchnorm(x)
        elif self.norm_layer == "instancenorm":
            normed = self.instancenorm(x)

        # Resnet1D
        out = self.resnet(normed)  # (batch_size, 512)
        out = self.flatten_d1(self.avg_pool_1d(out))
        ## Output necessary params
        raw = self.flatten_d0(self.signal_or_noise(out))
        pred_prob = self.sigmoid(raw)

        ## Parameter Estimation
        # Time of Coalescence
        tc = self.flatten_d0(self.coalescence_time(out))
        norm_tc = self.sigmoid(tc)
        # Chirp Distance
        dchirp = self.flatten_d0(self.chirp_distance(out))
        norm_dchirp = self.sigmoid(dchirp)
        # Chirp Mass
        mchirp = self.flatten_d0(self.chirp_mass(out))
        norm_mchirp = self.sigmoid(mchirp)
        # Distance
        dist = self.flatten_d0(self.distance(out))
        norm_dist = self.sigmoid(dist)
        # Mass Ratio
        q = self.flatten_d0(self.mass_ratio(out))
        norm_q = self.sigmoid(q)
        # Inverse Mass Ratio
        invq = self.flatten_d0(self.inv_mass_ratio(out))
        norm_invq = self.sigmoid(invq)
        # SNR
        snr = self.flatten_d0(self.snr(out))
        norm_snr = self.sigmoid(snr)

        # Return ouptut params (pred_prob, raw, cnn_output, pe_params)
        return {
            "raw": raw,
            "pred_prob": pred_prob,
            "input": x,
            "norm_tc": norm_tc,
            "norm_dchirp": norm_dchirp,
            "norm_mchirp": norm_mchirp,
            "norm_dist": norm_dist,
            "norm_q": norm_q,
            "norm_invq": norm_invq,
            "norm_snr": norm_snr,
            "tc": tc,
            "dchirp": dchirp,
            "mchirp": mchirp,
            "dist": dist,
            "q": q,
            "invq": invq,
            "snr": snr,
        }
