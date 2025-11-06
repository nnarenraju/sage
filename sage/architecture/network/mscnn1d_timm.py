#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : mscnn1d_timm.py
Description     : Short description of the file

Created on 2025-11-06 13:04:50

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


@unreviewed_model
class KappaModelPE(torch.nn.Module):
    """
    Kappa-type Model PE Architecture

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
        timm_params: dict = {
            "model_name": "resnet34",
            "pretrained": True,
            "in_chans": 2,
            "drop_rate": 0.25,
        },
        norm_layer: str = "layernorm",
        store_device: str = "cpu",
        **kwargs
    ):

        super().__init__()

        self.model_name = model_name
        self.store_device = store_device
        self.timm_params = timm_params
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.norm_layer = norm_layer

        """ Backend """
        # filters_start=16, kernel_start=32 --> 1.3 Mil. trainable params backend
        # filters_start=32, kernel_start=64 --> 9.6 Mil. trainable params backend
        self._det1 = ConvBlock(self.filter_size, self.kernel_size)
        self._det2 = ConvBlock(self.filter_size, self.kernel_size)
        _initialize_weights(self)

        """ Frontend """
        # resnet34 --> 21 Mil. trainable params trainable frontend
        self.frontend = timm.create_model(**timm_params)

        """ Mods """
        # Primary outputs
        self.signal_or_noise = nn.Linear(self.frontend.num_features, 1)
        self.coalescence_time = nn.Linear(self.frontend.num_features, 2)
        self.chirp_distance = nn.Linear(self.frontend.num_features, 1)
        self.chirp_mass = nn.Linear(self.frontend.num_features, 2)
        self.distance = nn.Linear(self.frontend.num_features, 1)
        self.mass_ratio = nn.Linear(self.frontend.num_features, 1)
        self.inv_mass_ratio = nn.Linear(self.frontend.num_features, 1)
        self.snr = nn.Linear(self.frontend.num_features, 2)
        # Manipulation layers
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.batchnorm = nn.BatchNorm1d(2)
        self.layernorm = nn.LayerNorm([2, 5830])
        self.layernorm_cnn = nn.LayerNorm([2, 128, 182])
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(self.frontend.num_features)
        self.flatten_d1 = nn.Flatten(start_dim=1)
        self.flatten_d0 = nn.Flatten(start_dim=0)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()

        ## Convert network into given dtype and store in proper device
        # Manipulation layers
        self.batchnorm.to(dtype=data_type, device=self.store_device)
        self.layernorm.to(dtype=data_type, device=self.store_device)
        self.layernorm_cnn.to(dtype=data_type, device=self.store_device)
        # Mod layers
        self.signal_or_noise.to(dtype=data_type, device=self.store_device)
        self.coalescence_time.to(dtype=data_type, device=self.store_device)
        self.chirp_distance.to(dtype=data_type, device=self.store_device)
        self.chirp_mass.to(dtype=data_type, device=self.store_device)
        self.distance.to(dtype=data_type, device=self.store_device)
        self.mass_ratio.to(dtype=data_type, device=self.store_device)
        self.inv_mass_ratio.to(dtype=data_type, device=self.store_device)
        self.snr.to(dtype=data_type, device=self.store_device)
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

        # Conv Backend
        cnn_output = torch.cat(
            [
                self.backend["det1"](normed[:, 0:1]),
                self.backend["det2"](normed[:, 1:2]),
            ],
            dim=1,
        )
        # Apply LayerNorm to CNN output before passing to ResNet
        cnn_output = self.layernorm_cnn(cnn_output)

        # Timm Frontend
        out = self.frontend(cnn_output)  # (batch_size, 1000) by default
        ## Manipulate encoder output to get params
        # Global Pool
        out = self.flatten_d1(self.avg_pool_1d(out))
        # In the Kaggle architecture a dropout is added at this point
        # I see no reason to include at this stage. But we can experiment.
        ## Output necessary params
        raw = self.flatten_d0(self.signal_or_noise(out))
        pred_prob = self.sigmoid(raw)
        # Parameter Estimation
        tc = self.flatten_d0(self.sigmoid(self.coalescence_time(out)))
        dchirp = self.flatten_d0(self.sigmoid(self.chirp_distance(out)))
        mchirp = self.flatten_d0(self.sigmoid(self.chirp_mass(out)))
        dist = self.flatten_d0(self.sigmoid(self.distance(out)))
        q = self.flatten_d0(self.sigmoid(self.mass_ratio(out)))
        invq = self.flatten_d0(self.sigmoid(self.inv_mass_ratio(out)))
        raw_snr = self.flatten_d0(self.snr(out))
        snr = self.sigmoid(raw_snr)

        # Return ouptut params (pred_prob, raw, cnn_output, pe_params)
        return {
            "raw": raw,
            "pred_prob": pred_prob,
            "cnn_output": cnn_output,
            "norm_tc": tc,
            "norm_dchirp": dchirp,
            "norm_mchirp": mchirp,
            "norm_dist": dist,
            "norm_q": q,
            "norm_invq": invq,
            "norm_snr": snr,
            "raw_snr": raw_snr,
            "input": x,
            "normed": normed,
        }
