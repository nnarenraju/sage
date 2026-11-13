#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Frequency-domain Planck tapers: low-frequency roll-on, high-frequency
roll-off and their band-pass product.

Filename        : taper.py
Description     : Short description of the file

Created on 2026-02-08 00:45:34

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

import torch


def _taper(x, width):
    """
    Planck taper ``1 / (1 + exp(z))`` of the distance from a boundary.

    0 at ``x = 0``, rising to 1 as ``x`` approaches ``width - 1``.

    Parameters
    ----------
    x : torch.Tensor, shape ``(B, F)``
        Distance from the boundary in bins, clamped to >= 1e-12.
    width : float or torch.Tensor
        Taper width in frequency bins.

    Returns
    -------
    torch.Tensor, shape ``(B, F)``
        Taper weights in ``[0, 1]``.
    """
    eps = 1e-12
    x = torch.clamp(x, min=eps)
    w = width - 1.0
    z = w / x + w / (x - w)
    return 1.0 / (1.0 + torch.exp(z))


def fd_low_freq_taper(f, f_min, df, width_bins):
    """
    Low-frequency roll-on taper in the frequency domain.

    0 up to ``f_min``, rising to 1 over ``width_bins`` frequency bins,
    and 1 above.

    Parameters
    ----------
    f : torch.Tensor
        Frequency array (Hz).
    f_min : float
        Frequency at which the taper starts rising.
    df : float
        Frequency bin spacing (Hz).
    width_bins : int
        Number of bins over which the taper rises from 0 to 1.

    Returns
    -------
    torch.Tensor
        Multiplicative taper weights, same shape as ``f``.
    """
    x = (f - f_min) / df
    w = width_bins - 1.0
    # Taper only in (0, w); 0 below, 1 above
    return torch.where(
        x <= 0,
        torch.zeros_like(x),
        torch.where(x >= w, torch.ones_like(x), _taper(x, width_bins)),
    )


def fd_high_freq_taper(f, f_cut, df, width_bins):
    """
    High-frequency roll-off taper in the frequency domain.

    1 up to ``f_cut - (width_bins - 1) * df``, falling to 0 over
    ``width_bins`` bins, and 0 from ``f_cut``.

    Parameters
    ----------
    f : torch.Tensor
        Frequency array (Hz).
    f_cut : float
        Frequency at which the taper completes its roll-off.
    df : float
        Frequency bin spacing (Hz).
    width_bins : int
        Number of bins over which the taper falls from 1 to 0.

    Returns
    -------
    torch.Tensor
        Multiplicative taper weights, same shape as ``f``.
    """
    x = (f_cut - f) / df
    w = width_bins - 1.0
    # Taper only in (0, w); 0 beyond the cut, 1 before the taper
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
    Band-pass taper: the product of the low and high tapers.

    0 outside ``(f_min, f_cut)`` and 1 inside the band away from its
    edges.

    Parameters
    ----------
    f : torch.Tensor
        Frequency array (Hz), shape ``(B, F)`` or ``(F,)``.
    f_min : float
        Lower frequency edge (Hz).
    f_cut : float
        Upper frequency cutoff (Hz).
    df : float
        Frequency bin spacing (Hz).
    low_width : int
        Taper width at the low-frequency edge, in bins.
    high_width : int
        Taper width at the high-frequency edge, in bins.

    Returns
    -------
    torch.Tensor
        Multiplicative taper weights, same shape as ``f``.
    """
    w_lo = fd_low_freq_taper(f, f_min, df, low_width)
    w_hi = fd_high_freq_taper(f, f_cut, df, high_width)
    return w_lo * w_hi
