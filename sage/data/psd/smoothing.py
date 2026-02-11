#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : smoothing.py
Description     : Short description of the file

Created on 2026-02-11 20:48:57

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
import numpy as np

from scipy.interpolate import UnivariateSpline


def smooth_psd_log_spline(freqs, psd, line_regions=None, smooth_factor=None):
    """
    Smooth a noisy PSD estimate using a spline in log-log space.

    Parameters
    ----------
    freqs : (F,) array
        Frequency array (must be > 0).
    psd : (F,) array
        PSD values (must be > 0).
    smooth_factor : float or None
        Smoothing strength. Larger = smoother.
        If None, an automatic heuristic is used.

    Returns
    -------
    psd_smooth : (F,) array
        Smoothed PSD (same shape as input).
    """

    # remove zero freq if present
    f = freqs
    p = psd

    logf = np.log(f)
    logp = np.log(p)

    if smooth_factor is None:
        # heuristic: proportional to number of points
        smooth_factor = len(logf) * 0.2

    # default weights
    weights = np.ones_like(logp)

    # upweight line regions if provided
    if line_regions is not None:
        for f_low, f_high in line_regions:
            line_mask = (f >= f_low) & (f <= f_high)
            weights[line_mask] = 2

    spline = UnivariateSpline(logf, logp, s=smooth_factor, w=weights)

    logp_smooth = spline(logf)
    psd_smooth = np.exp(logp_smooth)

    # put back DC if needed
    out = psd.copy()
    out = psd_smooth

    return out
