#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : math.py
Description     : Short description of the file

Created on 2026-01-21 07:59:05

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

# Packages
import torch
import numpy as np


def torch_linear_interp(x, xp, fp):
    """
    1D linear interpolation like jnp.interp.
    x:  Tensor (...,)
    xp: Tensor (N,) increasing
    fp: Tensor (N,)
    """
    # indices where elements should be inserted
    idx = torch.searchsorted(xp, x, right=True)

    idx = torch.clamp(idx, 1, xp.numel() - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]

    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0)


def torch_cubic_interp(x, xp, fp):
    """
    1D Cubic interpolation like scipy.interpolate.CubicSpline
    x:  Tensor (...,)
    xp: Tensor (N,) increasing
    fp: Tensor (N,)
    """
    idx = torch.searchsorted(xp, x, right=True)
    idx = idx.clamp(1, xp.numel() - 2)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    x2 = xp[idx + 1]

    y0 = fp[idx - 1]
    y1 = fp[idx]
    y2 = fp[idx + 1]

    # Finite-difference slopes
    m1 = (y1 - y0) / (x1 - x0)
    m2 = (y2 - y1) / (x2 - x1)

    t = (x - x1) / (x2 - x1)

    # Hermite basis
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)

    return h00 * y1 + h10 * (x2 - x1) * m1 + h01 * y2 + h11 * (x2 - x1) * m2


@torch.jit.script
def torch_cubic_interp_uniform(
    xs: torch.Tensor,
    y: torch.Tensor,
    x0: float,
    dx: float,
):
    """
    Fast cubic interpolation on a uniform grid (Catmull-Rom).

    Args:
        xs: (...,) query points
        y:  (N,) sampled values on uniform grid
        x0: grid start
        dx: grid spacing

    Returns:
        (...,) interpolated values
    """

    # Continuous index
    t = (xs - x0) / dx

    # Left index
    i = torch.floor(t).long()

    # Clamp to valid range
    i = torch.clamp(i, 1, y.shape[0] - 3)

    # Local coordinate
    u = t - i

    # Fetch points
    y0 = y[i - 1]
    y1 = y[i]
    y2 = y[i + 1]
    y3 = y[i + 2]

    # Catmull–Rom coefficients
    a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
    b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    c = -0.5 * y0 + 0.5 * y2
    d = y1

    return ((a * u + b) * u + c) * u + d


class Normalise:
    """
    Normalise the variable using known bounds

        For example, norm_tc = (tc - min_val)/(max_val - min_val)
        The values of max_val and min_val are provided
        to the class. obj.norm can be called during
        data generation to get normalised values of tc, if needed.

    """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def norm(self, val):
        # Return normalised value in [0, 1]
        return (val - self.min_val) / (self.max_val - self.min_val)

    def unnorm(self, val):
        # Return unnormalised value in original scale
        return (val * (self.max_val - self.min_val)) + self.min_val


class Standardise:
    """
    Standardise a variable to zero mean and unit variance.
    """

    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def norm(self, val):
        # Return zero mean unit variance output
        return (val - self.mean) / (self.std + self.eps)

    def unnorm(self, val):
        # Return original scale output
        return val * (self.std + self.eps) + self.mean
