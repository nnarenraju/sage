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


class LogSplineSmoothing:

    def __init__(
        self,
        smooth_factor=None,
        upweight_regions=None,
        return_coeffs=False,
        noise_low_frequency_cutoff=15.0,
    ):
        self.smooth_factor = smooth_factor
        self.upweight_regions = upweight_regions
        self.return_coeffs = return_coeffs
        self.noise_low_frequency_cutoff = noise_low_frequency_cutoff

    def smooth(self, freqs, psd, smooth_factor=None):
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

        # Floor before masking
        # float32 normal range is upto ~1e-38
        # float32 subnormal range is till ~1e-45
        # so we pick the smallest representable value > 0
        p = np.maximum(p, 1e-45)

        # define cutoff
        mask = f >= self.noise_low_frequency_cutoff

        # only use trusted region for fitting
        f = f[mask]
        p = p[mask]

        logf = np.log(f)
        logp = np.log(p)

        if smooth_factor is None and self.smooth_factor is None:
            # heuristic: proportional to number of points
            self.smooth_factor = len(logf) * 0.2
        elif smooth_factor:
            self.smooth_factor = smooth_factor

        # default weights
        weights = np.ones_like(logp)

        # upweight special regions in freq space
        if self.upweight_regions is not None:
            for f_low, f_high in self.upweight_regions:
                line_mask = (f >= f_low) & (f <= f_high)
                weights[line_mask] = 2

        # Number and placement of knots might different between PSDs
        # Use LSQUnivariateSpline if you want to keep these fixed
        spline = UnivariateSpline(logf, logp, s=self.smooth_factor, w=weights)

        logp_smooth = spline(logf)
        psd_smooth = np.exp(logp_smooth)

        # put back DC if needed
        out = psd.copy()
        out[mask] = psd_smooth

        return out
