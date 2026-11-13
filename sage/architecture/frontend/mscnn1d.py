# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Multi-scale 1D CNN frontend without attention: one detector's strain to
a 2D feature map for the ResNet backend.

Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Tue Nov 21 17:19:53 2021

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2021, Sage
__credits__     = nnarenraju
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = nagarajan@uni-potsdam.de
__status__      = ['inProgress', 'Archived', *inUsage*, 'Debugging']


Github Repository: NULL

Documentation


"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch.nn import MaxPool1d, BatchNorm1d


class Conv1dSame(nn.Conv1d):
    """
    1D convolution with ``padding='same'``.

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
        Unused; every convolution uses ``padding='same'``.
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


class ConvBlock(nn.Module):
    """
    Three-stage multi-scale 1D CNN frontend for one detector.

    Three stages of two :class:`ConcatBlockConv5`: kernels ``k``,
    ``k//2+1`` and a stride-8 max-pool; ``k//2+1``, ``k//4+1`` and a
    stride-4 max-pool; ``k//4+1`` twice.

    Parameters
    ----------
    filters_start : int
        Filters of the first stage, doubled in each later stage.
    kernel_start : int
        Base kernel size ``k``.
    in_channels : int
        Number of input channels.
    """

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
        """
        Frontend feature map.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, in_channels, L)``
            Strain of one detector.

        Returns
        -------
        torch.Tensor, shape ``(B, 1, 4 * filters_start, L // 32)``
            Feature map for the 2D backend.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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
