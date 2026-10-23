#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Anti-aliased downsampling (BlurPool).

Zhang, "Making Convolutional Networks Shift-Invariant Again", ICML 2019. Naive
strided pooling / strided convolution subsamples WITHOUT a low-pass first, so any
feature content above the new Nyquist rate aliases (folds down and corrupts the
feature maps) and the network becomes shift-variant. The fix: dense operation at
stride 1, then a fixed low-pass BLUR, then the strided subsample.

These modules implement the blur + subsample step: a depthwise (per-channel),
non-learnable binomial (Pascal) low-pass. Place one AFTER a stride-1 max-pool or
conv to anti-alias a downsampling by ``stride``. ``filt_size`` widens with the
stride so the low-pass cutoff roughly tracks the new Nyquist rate (a stride-2 pool
needs cutoff pi/2, a stride-8 pool needs pi/8, i.e. a wider filter).

NB: Sage's physics multirate stage is already anti-aliased (63-tap Kaiser); these
handle the LEARNED downsampling (frontend max-pools, ResNet stem + strided convs).
"""

from math import comb

import torch
import torch.nn as nn
import torch.nn.functional as F


def _binomial_kernel(n):
    """1D binomial (Pascal-row) low-pass of length ``n`` (unnormalised)."""
    return torch.tensor([comb(n - 1, k) for k in range(n)], dtype=torch.float32)


class BlurPool1d(nn.Module):
    """Depthwise binomial low-pass + stride-``stride`` subsample (1D).

    Parameters
    ----------
    channels : int
        Number of channels (the filter is applied per-channel / depthwise).
    stride : int
        Downsampling factor.
    filt_size : int or None
        Binomial filter length. ``None`` -> ``2*stride + 1`` (a wider low-pass for
        larger strides, so the cutoff roughly matches the new Nyquist rate).
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
        x = F.pad(x, (self.pad, self.pad), mode="reflect")
        return F.conv1d(x, self.filt, stride=self.stride, groups=self.channels)


class BlurPool2d(nn.Module):
    """Depthwise separable-binomial low-pass + stride-``stride`` subsample (2D).

    Filter = outer product of the 1D binomial kernel. ``filt_size`` defaults to 3
    (``[1,2,1]``), Zhang's small-stride recommendation; deep feature maps are
    small, so a compact filter keeps the reflect-pad well within bounds.
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
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        return F.conv2d(x, self.filt, stride=self.stride, groups=self.channels)
