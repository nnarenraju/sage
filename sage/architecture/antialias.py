#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Anti-aliased downsampling (BlurPool) for the networks' learned layers.

A depthwise, fixed binomial low-pass followed by the strided subsample,
placed after a stride-1 max-pool or convolution.
"""

from math import comb

import torch
import torch.nn as nn
import torch.nn.functional as F


def _binomial_kernel(n):
    """
    Binomial (Pascal-row) low-pass of length ``n``, unnormalised.

    Parameters
    ----------
    n : int
        Filter length.

    Returns
    -------
    torch.Tensor, shape ``(n,)``
        Binomial coefficients, float32.
    """
    return torch.tensor([comb(n - 1, k) for k in range(n)], dtype=torch.float32)


class BlurPool1d(nn.Module):
    """
    Depthwise binomial low-pass and stride-``stride`` subsample (1D).

    Parameters
    ----------
    channels : int
        Number of channels; the filter is applied per channel.
    stride : int
        Downsampling factor.
    filt_size : int or None
        Binomial filter length; None gives ``2*stride + 1``, so the
        cutoff widens with the stride.
    """

    def __init__(self, channels, stride, filt_size=None):
        super().__init__()
        if filt_size is None:
            filt_size = 2 * stride + 1
        a = _binomial_kernel(filt_size)
        a = a / a.sum()
        self.register_buffer("filt", a.view(1, 1, -1).repeat(channels, 1, 1))
        self.stride = stride
        self.channels = channels
        self.pad = filt_size // 2

    def forward(self, x):
        """
        Reflect-pad, low-pass and subsample along the last axis.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, C, L)``
            Input features.

        Returns
        -------
        torch.Tensor, shape ``(B, C, L_out)``
            Subsampled features, ``L_out = ceil(L / stride)`` for an
            odd ``filt_size``.
        """
        x = F.pad(x, (self.pad, self.pad), mode="reflect")
        return F.conv1d(x, self.filt, stride=self.stride, groups=self.channels)


class BlurPool2d(nn.Module):
    """
    Depthwise binomial low-pass and stride-``stride`` subsample (2D).

    The filter is the outer product of the 1D binomial kernel.

    Parameters
    ----------
    channels : int
        Number of channels; the filter is applied per channel.
    stride : int
        Downsampling factor along both axes.
    filt_size : int
        Binomial filter length; 3 gives ``[1, 2, 1]``.
    """

    def __init__(self, channels, stride, filt_size=3):
        super().__init__()
        a = _binomial_kernel(filt_size)
        k = torch.outer(a, a)
        k = k / k.sum()
        self.register_buffer(
            "filt", k.view(1, 1, filt_size, filt_size).repeat(channels, 1, 1, 1)
        )
        self.stride = stride
        self.channels = channels
        self.pad = filt_size // 2

    def forward(self, x):
        """
        Reflect-pad, low-pass and subsample along the last two axes.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, C, H, W)``
            Input features.

        Returns
        -------
        torch.Tensor, shape ``(B, C, H_out, W_out)``
            Subsampled features.
        """
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        return F.conv2d(x, self.filt, stride=self.stride, groups=self.channels)
