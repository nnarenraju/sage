#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Log-log spline smoothing of an amplitude spectral density.

Filename        : smoothing.py
Description     : Log-log spline smoothing of a spectral density.

Created on 2026-02-11 20:48:57

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

import numpy as np

from scipy.interpolate import UnivariateSpline


class LogSplineSmoothing:
    """
    Spectral smoother: a univariate spline fit in log-log space.

    Fits :class:`scipy.interpolate.UnivariateSpline` to log amplitude
    against log frequency above ``noise_low_frequency_cutoff``; bins
    below it are returned unchanged. In Sage the input is an ASD.

    Parameters
    ----------
    smooth_factor : float or None
        Spline smoothing condition ``s``; larger is smoother. If
        ``None``, ``0.2`` times the number of fitted points is used on
        the first call.
    upweight_regions : list of tuple of float or None
        ``(f_low, f_high)`` bands fitted with weight 2 instead of 1.
    return_coeffs : bool
        Stored but unused.
    noise_low_frequency_cutoff : float
        Frequency (Hz) below which bins are not fitted.
    """

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

    def smooth(self, freqs, asd, smooth_factor=None):
        """
        Smooth a spectrum with a spline in log-log space.

        A truthy ``smooth_factor`` replaces the stored one for this and
        later calls.

        Parameters
        ----------
        freqs : numpy.ndarray, shape ``(F,)``
            Frequencies (Hz); fitted bins must be > 0.
        asd : numpy.ndarray, shape ``(F,)``
            Spectral values, an ASD in Sage.
        smooth_factor : float or None
            Spline smoothing condition ``s``; ``None`` keeps the stored
            value.

        Returns
        -------
        numpy.ndarray, shape ``(F,)``
            Copy of ``asd`` with the fitted bins smoothed.
        """
        f = freqs
        p = asd

        # Floor before the log
        p = np.maximum(p, 1e-60)

        mask = f >= self.noise_low_frequency_cutoff

        # Fit only above the cutoff
        f = f[mask]
        p = p[mask]

        logf = np.log(f)
        logp = np.log(p)

        if smooth_factor is None and self.smooth_factor is None:
            self.smooth_factor = len(logf) * 0.2
        elif smooth_factor:
            self.smooth_factor = smooth_factor

        weights = np.ones_like(logp)

        if self.upweight_regions is not None:
            for f_low, f_high in self.upweight_regions:
                line_mask = (f >= f_low) & (f <= f_high)
                weights[line_mask] = 2

        # Knot number and placement can differ between spectra
        spline = UnivariateSpline(logf, logp, s=self.smooth_factor, w=weights)

        logp_smooth = spline(logf)
        asd_smooth = np.exp(logp_smooth)

        # Bins below the cutoff keep their input values
        out = asd.copy()
        out[mask] = asd_smooth

        return out
