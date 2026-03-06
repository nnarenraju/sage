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

# PACKAGES
import timm
import json
import torch
from torch import nn
from datetime import date

# Importing architecture snippets from zoo
from sage.architecture.backend.resnet2d_cbam import (
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
class Rigatoni_MS_ResNetCBAM_legacy(torch.nn.Module):
    """
    Kappa-type Model PE Architecture with ResNet and CBAM

    Description - consists of a 2-channel ConvBlock backend and a Timm model frontend
                  this Model-type can be used to test the Kaggle architectures

    Parameters
    ----------
    model_name  = 'simple' : string
        Simple NN model name for Frontend. Save model with this name as attribute.
    pretrained  = False : Bool
        Pretrained option for saved models
        If True, weights are stored under the model_name in saved_models dir
        If model name already exists, throws an error (safety)
    in_channels = 2 : int
        Number of input channels (number of detectors)
    out_channels = 2 : int
        Number of output channels (signal, noise)
    store_device = 'cpu' : str
        Storage device for network (NOTE: make sure data is also stored in the same device)
    weights_path = '' : str
        Absolute path to the weights.pt file. Used when pretrained == True

    """

    def __init__(
        self,
        model_name="trainable_backend_and_frontend",
        filter_size: int = 32,
        kernel_size: int = 64,
        resnet_size: int = 50,
        norm_layer: str = "instancenorm",
        upsample_factor: float = 1.0,
        parameter_estimation=(
            "norm_tc",
            "norm_mchirp",
        ),
        _input_length: int = 4254,  # 4254
        _decimated_bins=None,
        store_device: str = "cpu",
        **kwargs,
    ):

        super().__init__()

        self.model_name = model_name
        self.store_device = store_device
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.norm_layer = norm_layer
        self.upsample_factor = upsample_factor
        self._decimated_bins = _decimated_bins

        """ Backend """
        # filters_start=16, kernel_start=32 --> 1.3 Mil. trainable params backend
        # filters_start=32, kernel_start=64 --> 9.6 Mil. trainable params backend
        self._det1 = ConvBlock(self.filter_size, self.kernel_size)
        self._det2 = ConvBlock(self.filter_size, self.kernel_size)
        _initialize_weights(self)

        """ Frontend """
        # resnet50 --> Based on Res2Net blocks
        # Pretrained model is for 3-channels. We use 2 channels.
        if resnet_size == 50:
            self.frontend = resnet50_cbam(pretrained=False)
        elif resnet_size == 152:
            self.frontend = resnet152_cbam(pretrained=False)

        """ Mods """
        # Manipulation layers
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.batchnorm = nn.BatchNorm1d(2)
        self.layernorm = nn.LayerNorm([2, _input_length])
        # self.layernorm_cnn = nn.LayerNorm([2, 128, int(_input_length/32.)])
        self.instancenorm = nn.InstanceNorm1d(2, affine=True)
        self.flatten_d1 = nn.Flatten(start_dim=1)
        self.flatten_d0 = nn.Flatten(start_dim=0)
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(512)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.upsample = nn.Upsample(scale_factor=self.upsample_factor, mode="bicubic")

        ## Convert network into given dtype and store in proper device
        # Primary outputs
        self.signal_or_noise = nn.Linear(512, 1)
        self.coalescence_time = nn.Linear(512, 2)
        self.chirp_distance = nn.Linear(512, 2)
        self.chirp_mass = nn.Linear(512, 2)
        self.distance = nn.Linear(512, 2)
        self.mass_ratio = nn.Linear(512, 2)
        self.inv_mass_ratio = nn.Linear(512, 2)
        self.snr = nn.Linear(512, 2)
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
        self.layernorm.to(dtype=data_type, device=self.store_device)
        # self.layernorm_cnn.to(dtype=data_type, device=self.store_device)
        self.instancenorm.to(dtype=data_type, device=self.store_device)
        # Main layers
        self._det1.to(dtype=data_type, device=self.store_device)
        self._det2.to(dtype=data_type, device=self.store_device)
        self.backend = {"det1": self._det1, "det2": self._det2}
        self.frontend.to(dtype=data_type, device=self.store_device)

    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        if self.norm_layer == "batchnorm":
            normed = self.batchnorm(x)
        elif self.norm_layer == "layernorm":
            normed = self.layernorm(x)
        elif self.norm_layer == "instancenorm":
            normed = self.instancenorm(x)

        # Upsampling of input normed data
        if self.upsample_factor != 1.0:
            normed = torch.cat(
                [self.upsample(normed[:, 0:1]), self.upsample(normed[:, 1:2])], dim=1
            )

        # Conv Backend
        cnn_output = torch.cat(
            [
                self.backend["det1"](normed[:, 0:1]),
                self.backend["det2"](normed[:, 1:2]),
            ],
            dim=1,
        )

        # Timm Frontend
        out = self.frontend(cnn_output)  # (batch_size, 512)
        out = self.flatten_d1(self.avg_pool_1d(out))
        ## Output necessary params
        raw = self.flatten_d0(self.signal_or_noise(out))
        pred_prob = self.sigmoid(raw)

        ## Parameter Estimation
        # Time of Coalescence
        tc_ = self.coalescence_time(out)
        tc = self.flatten_d0(tc_[:, 0])
        norm_tc = self.sigmoid(tc)
        tc_var = self.Tanh(self.flatten_d0(tc_[:, 1]))
        # Chirp Distance
        dchirp_ = self.chirp_distance(out)
        dchirp = self.flatten_d0(dchirp_[:, 0])
        norm_dchirp = self.sigmoid(dchirp)
        dchirp_var = self.flatten_d0(dchirp_[:, 1])
        # Chirp Mass
        mchirp_ = self.chirp_mass(out)
        mchirp = self.flatten_d0(mchirp_[:, 0])
        norm_mchirp = self.sigmoid(mchirp)
        mchirp_var = self.Tanh(self.flatten_d0(mchirp_[:, 1]))
        # Distance
        dist_ = self.distance(out)
        dist = self.flatten_d0(dist_[:, 0])
        norm_dist = self.sigmoid(dist)
        dist_var = self.flatten_d0(dist_[:, 1])
        # Mass Ratio
        q_ = self.mass_ratio(out)
        q = self.flatten_d0(q_[:, 0])
        norm_q = self.sigmoid(q)
        q_var = self.flatten_d0(q_[:, 1])
        # Inverse Mass Ratio
        invq_ = self.inv_mass_ratio(out)
        invq = self.flatten_d0(invq_[:, 0])
        norm_invq = self.sigmoid(invq)
        invq_var = self.flatten_d0(invq_[:, 1])
        # SNR
        snr_ = self.snr(out)
        snr = self.flatten_d0(snr_[:, 0])
        norm_snr = self.sigmoid(snr)
        snr_var = self.flatten_d0(snr_[:, 1])

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
            "norm_tc_sigma": tc_var,
            "norm_mchirp_sigma": mchirp_var,
            "norm_snr_sigma": snr_var,
            "norm_q_sigma": q_var,
            "norm_invq_sigma": invq_var,
            "norm_dist_sigma": dist_var,
            "norm_dchirp_sigma": dchirp_var,
            "input": x,
            "normed": normed,
            "embedding": out,
        }
