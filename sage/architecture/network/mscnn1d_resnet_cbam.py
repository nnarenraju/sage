#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : mscnn1d_resnet_cbam.py
Description     : Short description of the file

Created on 2025-11-06 12:56:01

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
from sage.architecture.zoo.resnet_cbam import (
    resnet50_cbam,
    resnet152_cbam,
    resnet34_cbam,
)
from sage.architecture.zoo.res2net_v1b import (
    res2net101_v1b_26w_4s,
    res2net50_v1b_26w_4s,
    res2net152_v1b_26w_4s,
)
from sage.architecture.zoo.osnet1d import osnet_ain_custom as osnet1d
from sage.architecture.zoo.kaggle import ConvBlock, _initialize_weights
from sage.architecture.frontend import MSFeatureExtractor, MultiScaleBlock
from sage.architecture.zoo.resnet1d import resnet50, resnet101, resnet152

# Code Review
from sage.utils.decorators import unreviewed_model
from sage.utils.review import set_review_date

# Datatype for storage
data_type = torch.float32


class Rigatoni_MS_ResNetCBAM(torch.nn.Module):
    """
    Rigatoni-type model with multi-scale feature extractor && ResNet-CBAM

    Description - Consists of a MSFeatureExtractor frontend for each detector and a
                  ResNet-CBAM model backend. Capable of PE point estimate
                  regularisation.

    Parameters
    ----------


    """

    def __init__(
        self,
        model_name: str = "Rigatoni_MS_ResNetCBAM",
        scales: list = [1, 2, 4, 0.5, 0.25],
        blocks: list = [
            [MultiScaleBlock, MultiScaleBlock],
            [MultiScaleBlock, MultiScaleBlock],
            [MultiScaleBlock, MultiScaleBlock],
        ],
        out_channels: list = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes: list = [
            [64, 64 // 2 + 1],
            [64 // 2 + 1, 64 // 4 + 1],
            [64 // 4 + 1, 64 // 4 + 1],
        ],
        compression_factor: list = [8, 4, 0],
        in_channels: int = 1,
        resnet_size: int = 50,
        parameter_estimation: tuple = (),
        norm_layer: str = "instancenorm",
        store_device: str = "cpu",
        review: bool = False,
        **kwargs
    ):

        super().__init__()

        # Saving last review date
        if review:
            last_review_date = date.today()
            model_name = self.__class__.__name__
            parent_name = "models"
            # set_review_date(parent_name, model_name, last_review_date)

        self.model_name = model_name
        self.norm_layer = norm_layer
        self.parameter_estimation = parameter_estimation
        self.store_device = store_device

        """ Backend """
        # Initialisation of weights and biases performed upon call
        self._det1 = MSFeatureExtractor(
            scales,
            blocks,
            out_channels,
            base_kernel_sizes,
            compression_factor,
            in_channels,
        )
        self._det2 = MSFeatureExtractor(
            scales,
            blocks,
            out_channels,
            base_kernel_sizes,
            compression_factor,
            in_channels,
        )

        """ Frontend """
        # Pretrained model is for 3-channels. We use 2 channels.
        # When training  on HLV, we can use pretrained model.
        if resnet_size == 50:
            self.backend = resnet50_cbam(pretrained=False)
        elif resnet_size == 152:
            self.backend = resnet152_cbam(pretrained=False)

        """ Mods """
        # Normalisation layers
        self.batchnorm = nn.BatchNorm1d(2)
        self.instancenorm = nn.InstanceNorm1d(2, affine=True)
        # Shape manipulation
        self.flatten_d1 = nn.Flatten(start_dim=1)
        self.flatten_d0 = nn.Flatten(start_dim=0)
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(512)
        # Value transformation
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        # Regularisation layers
        self.dropout = nn.Dropout(0.25)

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
        self._det1.to(dtype=data_type, device=self.store_device)
        self._det2.to(dtype=data_type, device=self.store_device)
        self.frontend = {"det1": self._det1, "det2": self._det2}
        self.backend.to(dtype=data_type, device=self.store_device)

    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        if self.norm_layer == "batchnorm":
            normed = self.batchnorm(x)
        elif self.norm_layer == "instancenorm":
            normed = self.instancenorm(x)

        # 1D CNN Frontend
        cnn_output = torch.cat(
            [
                self.frontend["det1"](normed[:, 0:1]),
                self.frontend["det2"](normed[:, 1:2]),
            ],
            dim=1,
        )

        # ResNet CBAM Backend
        out = self.backend(cnn_output)  # (batch_size, embedding_size)
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
            "cnn_output": cnn_output,
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
            "input": x,
            "normed": normed,
        }
