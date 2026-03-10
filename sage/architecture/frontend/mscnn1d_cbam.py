#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : mscnn1d_cbam.py
Description     : Short description of the file

Created on 2026-03-10 02:01:49

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


class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, T)
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        out = avg_out + max_out
        out = self.sigmoid(out).unsqueeze(-1)
        return x * out


class TemporalAttention1D(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, T)
        avg_out = x.mean(dim=1, keepdim=True)  # (B,1,T)
        max_out, _ = x.max(dim=1, keepdim=True)  # (B,1,T)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


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

        self.ca1 = ChannelAttention1D(filters_start)
        self.ta1 = TemporalAttention1D(filters_start)

        self.conv2 = nn.Sequential(
            ConcatBlockConv5(filters_start, filters_start * 2, k2, bias=False),
            ConcatBlockConv5(filters_start * 2, filters_start * 2, k3, bias=False),
            MaxPool1d(kernel_size=4, stride=4),
        )

        self.ca2 = ChannelAttention1D(filters_start * 2)
        self.ta2 = TemporalAttention1D(filters_start * 2)

        self.conv3 = nn.Sequential(
            ConcatBlockConv5(filters_start * 2, filters_start * 4, k3, bias=False),
            ConcatBlockConv5(filters_start * 4, filters_start * 4, k3, bias=False),
        )

        self.ca3 = ChannelAttention1D(filters_start * 4)
        self.ta3 = TemporalAttention1D(filters_start * 4)

    def forward(self, x):

        x = self.conv1(x)
        x = self.ca1(x)
        x = self.ta1(x)

        x = self.conv2(x)
        x = self.ca2(x)
        x = self.ta2(x)

        x = self.conv3(x)
        x = self.ca3(x)
        x = self.ta3(x)

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
