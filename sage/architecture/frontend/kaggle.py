# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Tue Nov 21 17:19:53 2021

__author__      = nnarenraju
__copyright__   = Copyright 2021, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation


"""

# Future imports
from __future__ import annotations

# Modules
import math

from collections.abc import Callable

# PyTorch imports
import torch
import torch.nn as nn

from torch import conv1d
from typing import Optional
from torch.nn import MaxPool1d, BatchNorm1d
from torch.nn.functional import pad



""" 
Kaggle G2NET (2021) 1st Place Architecture
Credits to:
    1. Denis Kanonik (kaggle master)
    2. Selim Seferkov (kaggle grandmaster)

Modified for MLMDC1 usage: Narenraju Nagarajan (PGR - UofG)

Calculating the output shape:
        output_dim = ((W-K+2.0*P)/S)+1
        W - Input volume
        K - Kernel size
        P - Padding
        S - Stride
        (Round-down if the result is not an integer)
    
    Example for G2NET dataset:
        2 second time-series sampled at 2048 Hz
        W = 4096 samples
        K = 5 kernel size
        P = (5 - 1)//2 = 2
        S = 4
        output_dim = ((4096-5+2.0*2)/4.0)+1 = 1024.75 = 1024 (round-down)
        output_shape = (3 channels, 128 out_channels, 1024 output_dim)
                     = (3, 128, 1024) for each batch
    
    Stride Selection for down-sampling:
        0.1 seconds = (1/10) * 2048 = 204.8 samples
        To keep loss to a minium, set stride = 2 (max.)
        Keep stride = 1 to preserve input shape
        If padding == "same", no striding is allowed
    
    Working with MaxPool1d:
        Input size = (N, C, L)
        Output size = (N, C, Lout)
        
        Lout = ((Lin + 2.0*padding - dilation * (kernel_size - 1) - 1)/stride) + 1
    
    Link to Weight-Initialisation Techniques:
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    TL;DR:
        [1] Use Xavier and normalised Xavier for Sigmoid and Tanh activation
        [2] Use Kaiming Normal (aka. He Weight Initialisation) for ReLU activation

"""

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
        return conv1d_same(x, self.weight, self.bias, self.stride[0], self.padding[0], self.dilation[0], self.groups)

class ConcatBlockConv5(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, act=nn.SiLU):
        super().__init__()
        self.c1 = nn.Sequential(
            Conv1dSame(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            BatchNorm1d(out_channels),
            act())
        self.c2 = nn.Sequential(
            Conv1dSame(in_channels, out_channels, kernel_size * 2, stride, padding, dilation, groups, bias),
            BatchNorm1d(out_channels),
            act())
        self.c3 = nn.Sequential(
            Conv1dSame(in_channels, out_channels, kernel_size // 2, stride, padding, dilation, groups, bias),
            BatchNorm1d(out_channels),
            act())
        self.c4 = nn.Sequential(
            Conv1dSame(in_channels, out_channels, kernel_size // 4, stride, padding, dilation, groups, bias),
            BatchNorm1d(out_channels),
            act())
        self.c5 = nn.Sequential(
            Conv1dSame(in_channels, out_channels, kernel_size * 4, stride, padding, dilation, groups, bias),
            BatchNorm1d(out_channels),
            act())
        self.c6 = nn.Sequential(
            Conv1dSame(out_channels * 5 + in_channels, out_channels, 1, stride, padding, dilation, groups, bias),
            BatchNorm1d(out_channels),
            act())

    def forward(self, x):
        x = torch.cat([self.c1(x), self.c2(x), self.c3(x), self.c4(x), self.c5(x), x], dim=1)
        x = self.c6(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, filters_start=32, kernel_start=64, in_channels=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            ConcatBlockConv5(in_channels, filters_start, kernel_start, bias=False),
            ConcatBlockConv5(filters_start, filters_start, kernel_start // 2 + 1, bias=False),
            MaxPool1d(kernel_size=8, stride=8)
        )
        self.conv2 = nn.Sequential(
            ConcatBlockConv5(filters_start, filters_start * 2, kernel_start // 2 + 1,
                             bias=False),
            ConcatBlockConv5(filters_start * 2, filters_start * 2, kernel_start // 4 + 1, bias=False),
            MaxPool1d(kernel_size=4, stride=4)
        )
        self.conv3 = nn.Sequential(
            ConcatBlockConv5(filters_start * 2, filters_start * 4, kernel_start // 4 + 1, bias=False),
            ConcatBlockConv5(filters_start * 4, filters_start * 4, kernel_start // 4 + 1, bias=False),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x).unsqueeze(1)
        return x

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