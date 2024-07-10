#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Fri Jun 30 22:02:45 2023

__author__      = nnarenraju
__copyright__   = Copyright 2022, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

# Modules
import math

# PyTorch imports
import torch
import torch.nn as nn
from torch import conv1d
from typing import Optional
from torch.nn import functional as F
from torch.nn import MaxPool1d, BatchNorm1d
from torch.nn.functional import pad



# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: int, s: int, d: int = 1, value: float = 0):
    iw = x.size()[-1]
    pad_w = get_same_padding(iw, k, s, d)
    if pad_w > 0:
        x = pad(x, [pad_w // 2, pad_w - pad_w // 2], value=value)
    return x


def conv1d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: int = 1,
        padding=0, dilation: int = 1, groups: int = 1):
    x = pad_same(x, weight.shape[-1], stride, dilation)
    return conv1d(x, weight, bias, stride, 0, dilation, groups)


class Conv1dSame(nn.Conv1d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv1dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv1d_same(x, self.weight, self.bias, self.stride[0], 
               self.padding[0], self.dilation[0], self.groups)


class FCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0, 
                 dilation=1, groups=1, bias=False, activation=nn.SiLU):
        super().__init__()
        self.block = nn.Sequential(
            Conv1dSame(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm1d(out_channels),
            activation(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class FCN(nn.Module):
    def __init__(self, device, _input_length):
        super().__init__()

        self.norm = nn.LayerNorm([2, 2048]) 

        self.feature_extractor = nn.Sequential(
            FCNBlock(2, 64),
            FCNBlock(64, 64),
            FCNBlock(64, 64),
            FCNBlock(64, 64),
            FCNBlock(64, 128, stride=2),
            FCNBlock(128, 128),
            FCNBlock(128, 128),
            FCNBlock(128, 256, stride=2),
            FCNBlock(256, 256),
            FCNBlock(256, 256),
            FCNBlock(256, 256, stride=2),
            FCNBlock(256, 256),
            FCNBlock(256, 256),
            FCNBlock(256, 512, stride=2),
            FCNBlock(512, 512),
            FCNBlock(512, 512),
            FCNBlock(512, 512, stride=2),
            FCNBlock(512, 512),
            FCNBlock(512, 256),
            FCNBlock(256, 256),
            FCNBlock(256, 128),
            FCNBlock(128, 128),
            FCNBlock(128, 128),
            FCNBlock(128, 64),
            FCNBlock(64, 64),
            FCNBlock(64, 64),
            FCNBlock(64, 64),
        )
        self.reduce = nn.Sequential(
            nn.AdaptiveAvgPool2d((64, 1)),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            Conv1dSame(64, 1, 1)
        )
        
        _initialize_weights(self)
        self.norm.to(dtype=torch.float32, device=device)
        self.feature_extractor.to(dtype=torch.float32, device=device)
        self.reduce.to(dtype=torch.float32, device=device)

    def forward(self, x):
        normed = self.norm(x)
        embedding = self.feature_extractor(normed)
        out = self.reduce(embedding).squeeze()
        return out


def _initialize_weights(self):
    # Initialising weights to all layers
    for m in self.modules():
        if isinstance(m, nn.Conv1d):
            m.weight.data = nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
