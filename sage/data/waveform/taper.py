#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : taper.py
Description     : Short description of the file

Created on 2026-02-08 00:45:34

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


def _taper(x, width):
    """
    x: distance from boundary (>= 0), shape (B, F)
    width: taper width in bins (scalar or (B,1))
    """
    eps = 1e-12
    x = torch.clamp(x, min=eps)
    w = width - 1.0
    z = w / x + w / (x - w)
    return 1.0 / (1.0 + torch.exp(z))


def fd_low_freq_taper(f, f_min, df, width_bins):
    x = (f - f_min) / df
    w = width_bins - 1.0
    # Apply formula only in (0, w), 0 below, 1 above
    return torch.where(
        x <= 0,
        torch.zeros_like(x),
        torch.where(x >= w, torch.ones_like(x), _taper(x, width_bins)),
    )


def fd_high_freq_taper(f, f_cut, df, width_bins):
    x = (f_cut - f) / df
    w = width_bins - 1.0
    # Apply formula only in (0, w), 0 beyond cut, 1 before taper start
    return torch.where(
        x <= 0,
        torch.zeros_like(x),
        torch.where(x >= w, torch.ones_like(x), _taper(x, width_bins)),
    )


def fd_taper(
    f,
    f_min,
    f_cut,
    df,
    low_width=64,
    high_width=64,
):
    """
    Returns multiplicative taper of shape (B, F)
    """
    w_lo = fd_low_freq_taper(f, f_min, df, low_width)
    w_hi = fd_high_freq_taper(f, f_cut, df, high_width)
    return w_lo * w_hi
