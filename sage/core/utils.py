#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : utils.py
Description     : Short description of the file

Created on 2025-11-06 17:39:59

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

# Packages
import torch
import numpy as np


def to_sequence(x):
    """Wrap scalars into a tuple so everything becomes iterable."""
    if x is None:
        return x
    if isinstance(x, (str, bytes)):
        return (x,)  # single string as a tuple
    if isinstance(x, (int, float)):
        return (x,)
    if isinstance(x, (bool)):
        return (x,)
    return tuple(x)


def ensure_1d(x):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input must be 1D")
    return x


def torch_interp(x, xp, fp):
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
    Standardise the variable to zero mean and unit variance
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
