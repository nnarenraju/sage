# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Tue Nov 21 17:19:53 2021

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2021, Sage
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = nagarajan@uni-potsdam.de
__status__      = ['inProgress', 'Archived', *inUsage*, 'Debugging']


Github Repository: NULL

Documentation


"""

# Future imports
from __future__ import annotations

# PyTorch imports
import torch
import torch.nn as nn

from torch.nn import MaxPool1d, BatchNorm1d


class Conv1dSame(nn.Conv1d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding="same",
            dilation=dilation,
            groups=groups,
            bias=bias,
        )


class ConcatBlockConv5(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        act=nn.SiLU,
    ):
        super().__init__()

        k1 = kernel_size
        k2 = kernel_size * 2
        k3 = kernel_size // 2
        k4 = kernel_size // 4
        k5 = kernel_size * 4

        self.c1 = nn.Sequential(
            Conv1dSame(
                in_channels,
                out_channels,
                k1,
                stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            BatchNorm1d(out_channels),
            act(inplace=True),
        )

        self.c2 = nn.Sequential(
            Conv1dSame(
                in_channels,
                out_channels,
                k2,
                stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            BatchNorm1d(out_channels),
            act(inplace=True),
        )

        self.c3 = nn.Sequential(
            Conv1dSame(
                in_channels,
                out_channels,
                k3,
                stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            BatchNorm1d(out_channels),
            act(inplace=True),
        )

        self.c4 = nn.Sequential(
            Conv1dSame(
                in_channels,
                out_channels,
                k4,
                stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            BatchNorm1d(out_channels),
            act(inplace=True),
        )

        self.c5 = nn.Sequential(
            Conv1dSame(
                in_channels,
                out_channels,
                k5,
                stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            BatchNorm1d(out_channels),
            act(inplace=True),
        )

        self.c6 = nn.Sequential(
            Conv1dSame(
                out_channels * 5 + in_channels,
                out_channels,
                1,
                stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            BatchNorm1d(out_channels),
            act(inplace=True),
        )

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        x5 = self.c5(x)

        x = torch.cat((x1, x2, x3, x4, x5, x), dim=1)

        return self.c6(x)


class ConvBlock(nn.Module):
    def __init__(self, filters_start=32, kernel_start=64, in_channels=1):
        super().__init__()

        k1 = kernel_start
        k2 = kernel_start // 2 + 1
        k3 = kernel_start // 4 + 1

        self.conv1 = nn.Sequential(
            ConcatBlockConv5(in_channels, filters_start, k1, bias=False),
            ConcatBlockConv5(filters_start, filters_start, k2, bias=False),
            MaxPool1d(kernel_size=8, stride=8),
        )

        self.conv2 = nn.Sequential(
            ConcatBlockConv5(filters_start, filters_start * 2, k2, bias=False),
            ConcatBlockConv5(filters_start * 2, filters_start * 2, k3, bias=False),
            MaxPool1d(kernel_size=4, stride=4),
        )

        self.conv3 = nn.Sequential(
            ConcatBlockConv5(filters_start * 2, filters_start * 4, k3, bias=False),
            ConcatBlockConv5(filters_start * 4, filters_start * 4, k3, bias=False),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.unsqueeze(1)
        return x


def _initialize_frontend_weights(self):

    for m in self.modules():

        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
