#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : simple.py
Description     : Short description of the file

Created on 2025-11-06 12:49:16

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
class GammaModel(torch.nn.Module):
    """
    Gamma-type Model Architecture

    Description - consists of a 2-channel simple NN frontend (no backend)

    Parameters
    ----------
    model_name  = 'simple' : string
        Simple NN model name for Frontend. Save model with this name as attribute.
    in_channels = 2 : int
        Number of input channels (number of detectors)
    out_channels = 2 : int
        Number of output channels (signal, noise)
    store_device = 'cpu' : str
        Storage device for network (NOTE: make sure data is also stored in the same device)

    """

    def __init__(
        self,
        model_name="simple",
        in_channels: int = 2,
        out_channels: int = 2,
        flatten_size: int = 1088,
        store_device: str = "cpu",
    ):

        super().__init__()

        self.model_name = model_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flatten_size = flatten_size
        self.store_device = store_device

        # Initialise Frontend Model
        # Add the following line as last layer if softmax is needed
        # torch.nn.Softmax(dim=1) --> (signal, noise)
        self.frontend = torch.nn.Sequential(  # Shapes
            torch.nn.BatchNorm1d(self.in_channels),  #  2x2048
            torch.nn.Conv1d(2, 4, 64),  #  4x1985
            torch.nn.ELU(),  #  4x1985
            torch.nn.Conv1d(4, 4, 32),  #  4x1954
            torch.nn.MaxPool1d(4),  #  4x 489
            torch.nn.ELU(),  #  4x 489
            torch.nn.Conv1d(4, 8, 32),  #  8x 458
            torch.nn.ELU(),  #  8x 458
            torch.nn.Conv1d(8, 8, 16),  #  8x 443
            torch.nn.MaxPool1d(3),  #  8x 147
            torch.nn.ELU(),  #  8x 147
            torch.nn.Conv1d(8, 16, 16),  # 16x 132
            torch.nn.ELU(),  # 16x 132
            torch.nn.Conv1d(16, 16, 16),  # 16x 117
            torch.nn.MaxPool1d(4),  # 16x  29
            torch.nn.ELU(),  # 16x  29
            torch.nn.Flatten(),  #      xx
            torch.nn.Linear(self.flatten_size, 32),  #      32
            torch.nn.Dropout(p=0.5),  #      32
            torch.nn.ELU(),  #      32
            torch.nn.Linear(32, 16),  #      16
            torch.nn.Dropout(p=0.5),  #      16
            torch.nn.ELU(),  #      16
            torch.nn.Linear(16, self.out_channels),  #       2/1
        )

        # Convert network into given dtype and store in proper device
        self.frontend.to(dtype=data_type, device=self.store_device)
        self.sigmoid = torch.nn.Sigmoid()

    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        # Simple NN frontend (no backend)
        raw = self.frontend(x)
        pred_prob = self.sigmoid(raw)
        return {"pred_prob": pred_prob, "raw": raw}
