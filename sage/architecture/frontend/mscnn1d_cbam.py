#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-scale 1D CNN frontend with channel and temporal attention: one
detector's strain to a 2D feature map for the ResNet backend.

Filename        : mscnn1d_cbam.py
Description     : Short description of the file

Created on 2026-03-10 02:01:49

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch.nn import MaxPool1d, BatchNorm1d

from ..antialias import BlurPool1d


class Conv1dSame(nn.Conv1d):
    """
    1D convolution with ``padding="same"``.

    Parameters
    ----------
    in_channels, out_channels : int
        Input and output channels.
    kernel_size : int
        Kernel size.
    stride, dilation, groups : int
        As in :class:`torch.nn.Conv1d`.
    bias : bool
        Whether the convolution has a bias.
    """

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
    """
    Inception-style block of five parallel 1D convolutions.

    Branches at kernel sizes ``k``, ``2k``, ``k//2``, ``k//4`` and
    ``4k`` (each Conv-BN-activation) are concatenated with the input
    and fused by a 1x1 Conv-BN-activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Output channels of each branch and of the fused output.
    kernel_size : int
        Base kernel size ``k``.
    stride : int
        Stride of every convolution.
    padding : int
        Unused; every convolution uses ``padding="same"``.
    dilation, groups : int
        As in :class:`torch.nn.Conv1d`, for every convolution.
    bias : bool
        Whether the convolutions have a bias.
    act : type
        Activation class, instantiated with ``inplace=True``.
    """

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
        """
        Fused multi-scale features.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, in_channels, L)``
            Input features.

        Returns
        -------
        torch.Tensor, shape ``(B, out_channels, L)``
            Output features.
        """
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        x5 = self.c5(x)

        x = torch.cat((x1, x2, x3, x4, x5, x), dim=1)

        return self.c6(x)


class ChannelAttention1D(nn.Module):
    """
    1D CBAM channel attention gate.

    Average- and max-pooled channel descriptors pass through a shared
    two-layer MLP; the sigmoid of their sum rescales each channel.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    reduction : int
        Bottleneck reduction ratio of the MLP.
    """

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
        """
        Channel-gated features.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, C, T)``
            Input features.

        Returns
        -------
        torch.Tensor, shape ``(B, C, T)``
            ``x`` scaled by the channel weights.
        """
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        out = avg_out + max_out
        out = self.sigmoid(out).unsqueeze(-1)
        return x * out


class TemporalAttention1D(nn.Module):
    """
    1D CBAM temporal attention gate.

    The channel-wise mean and max pass through one convolution and a
    sigmoid; the result rescales every time step.

    Parameters
    ----------
    in_channels : int
        Unused; the convolution sees the two pooled channels.
    kernel_size : int
        Kernel size of the attention convolution.
    """

    def __init__(self, in_channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Time-gated features.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, C, T)``
            Input features.

        Returns
        -------
        torch.Tensor, shape ``(B, C, T)``
            ``x`` scaled by the temporal weights.
        """
        avg_out = x.mean(dim=1, keepdim=True)  # (B,1,T)
        max_out, _ = x.max(dim=1, keepdim=True)  # (B,1,T)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


class ConvBlock(nn.Module):
    """
    Three-stage multi-scale 1D CNN frontend with CBAM attention.

    Each stage: two :class:`ConcatBlockConv5` with kernels from ``k``
    down to ``k//4+1``, a downsample (by 8, by 4, none), channel and
    temporal attention, and channel dropout.

    Parameters
    ----------
    filters_start : int
        Filters of the first stage, doubled in each later stage.
    kernel_start : int
        Base kernel size ``k``.
    in_channels : int
        Number of input channels.
    dropout : float
        Channel dropout rate after each stage.
    use_blurpool : bool
        Downsample by a stride-1 max-pool then a BlurPool, instead of
        a strided max-pool.
    """

    def __init__(self, filters_start=32, kernel_start=64, in_channels=1, dropout=0.0,
                 use_blurpool=True):
        super().__init__()

        # Dropout1d zeroes whole channels
        self.drop1 = nn.Dropout1d(dropout)
        self.drop2 = nn.Dropout1d(dropout)
        self.drop3 = nn.Dropout1d(dropout)

        k1 = kernel_start
        k2 = kernel_start // 2 + 1
        k3 = kernel_start // 4 + 1

        # Downsample by 8: stride-1 max-pool then BlurPool, or a
        # stride-8 max-pool
        _down1 = (
            [MaxPool1d(kernel_size=8, stride=1, padding=4),
             BlurPool1d(filters_start, stride=8)]
            if use_blurpool else
            [MaxPool1d(kernel_size=8, stride=8)]
        )
        self.conv1 = nn.Sequential(
            ConcatBlockConv5(in_channels, filters_start, k1, bias=False),
            ConcatBlockConv5(filters_start, filters_start, k2, bias=False),
            *_down1,
        )

        self.ca1 = ChannelAttention1D(filters_start)
        self.ta1 = TemporalAttention1D(filters_start)

        # Downsample by 4, as above
        _down2 = (
            [MaxPool1d(kernel_size=4, stride=1, padding=2),
             BlurPool1d(filters_start * 2, stride=4)]
            if use_blurpool else
            [MaxPool1d(kernel_size=4, stride=4)]
        )
        self.conv2 = nn.Sequential(
            ConcatBlockConv5(filters_start, filters_start * 2, k2, bias=False),
            ConcatBlockConv5(filters_start * 2, filters_start * 2, k3, bias=False),
            *_down2,
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
        """
        Frontend feature map.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, in_channels, L)``
            Strain of one detector.

        Returns
        -------
        torch.Tensor, shape ``(B, 1, 4 * filters_start, L_down)``
            Feature map for the 2D backend; ``L_down`` is about
            ``L/32``.
        """

        x = self.conv1(x)
        x = self.ca1(x)
        x = self.ta1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.ca2(x)
        x = self.ta2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.ca3(x)
        x = self.ta3(x)
        x = self.drop3(x)

        x = x.unsqueeze(1)
        return x


def _initialize_frontend_weights(self):
    """
    Initialise the Conv1d, BatchNorm1d and Linear layers of a module.

    Kaiming-normal Conv1d weights, unit BatchNorm1d scales, normal
    Linear weights with std 0.01, and zero biases.

    Parameters
    ----------
    self : nn.Module
        Module whose parameters are initialised in place.
    """

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
