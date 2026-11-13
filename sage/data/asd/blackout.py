#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Blackout policies: inflate selected bins of the median ASD to
suppress spectral lines and glitches.

Filename        : blackout.py
Description     : Short description of the file

Created on 2026-01-20 12:47:25

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


class BlackoutPolicy:
    """
    Base class of the ASD blackout policies.

    A policy inflates selected bins of the median ASD so they contribute
    little to the whitened data. Ratios are amplitude ratios of the ASDs
    from :meth:`~sage.data.asd.get_asds.EstimateASD.estimate_raw_asds`:
    a ``max_ratio`` of 4 is a factor 16 in power.
    """

    def apply(self, median_asd, max_asd):
        """
        Apply the blackout policy to the median ASD.

        The base policy returns the median ASD unchanged.

        Parameters
        ----------
        median_asd : numpy.ndarray, shape ``(F,)``
            Median ASD over the sampled noise.
        max_asd : numpy.ndarray, shape ``(F,)``
            Per-bin maximum ASD over the sampled noise.

        Returns
        -------
        asd : numpy.ndarray, shape ``(F,)``
            ASD, inflated at the blackout bins.
        idxs : numpy.ndarray of int or None
            Indices of the modified bins; ``None`` here.
        """
        return median_asd, None


class HardRatioBlackout(BlackoutPolicy):
    """
    Set bins where ``max_asd / median_asd > max_ratio`` to ``1e12``.

    Parameters
    ----------
    max_ratio : float
        Ratio above which a bin is blacked out.
    """

    def __init__(self, max_ratio):
        self.max_ratio = max_ratio

    def apply(self, median_asd, max_asd):
        """
        Set bins above the ratio threshold to ``1e12``.

        Parameters and returns as in :meth:`BlackoutPolicy.apply`.
        """
        ratio = max_asd / median_asd
        idxs = np.where(ratio > self.max_ratio)[0]

        asd = median_asd.copy()
        asd[idxs] = 1e12

        # Fraction of bins blacked out; not used
        blacked_out_frac = len(idxs) / len(ratio)

        return asd, idxs


class SoftRatioBlackout(BlackoutPolicy):
    """
    Power-law inflation of bins whose ASD ratio exceeds a threshold.

    The median ASD is multiplied by
    ``1 + alpha * ((ratio / max_ratio)**beta - 1)`` at the bins where
    ``ratio = max_asd / median_asd > max_ratio``.

    Parameters
    ----------
    max_ratio : float
        Ratio above which inflation begins.
    alpha : float
        Strength of the inflation.
    beta : float
        Power-law exponent.
    max_scale : float or None
        Cap on the inflation factor.
    """

    def __init__(
        self,
        max_ratio: float,
        alpha: float = 5.0,
        beta: float = 2.0,
        max_scale: float | None = None,
    ):
        self.max_ratio = max_ratio
        self.alpha = alpha
        self.beta = beta
        self.max_scale = max_scale

    def apply(self, median_asd, max_asd):
        """
        Inflate bins above the ratio threshold by a power law.

        Parameters and returns as in :meth:`BlackoutPolicy.apply`.
        """
        ratio = max_asd / median_asd

        scale = np.ones_like(median_asd)

        mask = ratio > self.max_ratio
        scale[mask] += self.alpha * ((ratio[mask] / self.max_ratio) ** self.beta - 1)

        if self.max_scale is not None:
            scale = np.minimum(scale, self.max_scale)

        return median_asd * scale, np.where(mask)[0]


class GaussianSoftNotchBlackout(BlackoutPolicy):
    """
    Inflate the ASD at given line frequencies with Gaussian bumps.

    The median ASD is multiplied by
    ``1 + depth * sum(exp(-0.5 * ((freqs - center) / width)**2))``.

    Parameters
    ----------
    freqs : array-like, shape ``(F,)``
        Frequencies of the ASD bins (Hz).
    centers : list of float
        Centre frequencies of the Gaussians (Hz).
    widths : list of float
        Standard deviations of the Gaussians (Hz).
    depth : float
        Peak inflation added by each Gaussian.
    """

    def __init__(self, freqs, centers, widths, depth=10.0):
        self.freqs = freqs
        self.centers = centers
        self.widths = widths
        self.depth = depth

    def apply(self, median_asd, max_asd):
        """
        Inflate the ASD with the Gaussian bumps; ``max_asd`` is unused.

        Parameters and returns as in :meth:`BlackoutPolicy.apply`; the
        returned indices are empty.
        """
        scale = np.ones_like(median_asd)

        for f0, w in zip(self.centers, self.widths):
            scale += self.depth * np.exp(-0.5 * ((self.freqs - f0) / w) ** 2)

        return median_asd * scale, np.empty(0, dtype=np.int64)


class LogSoftRatioBlackout(BlackoutPolicy):
    """
    Logarithmic inflation of bins whose ASD ratio exceeds a threshold.

    The median ASD is multiplied by
    ``1 + alpha * log1p((ratio - max_ratio) / max_ratio)`` at the bins
    where ``ratio = max_asd / median_asd > max_ratio``.

    Parameters
    ----------
    max_ratio : float
        Ratio above which inflation begins.
    alpha : float
        Strength of the inflation.
    max_scale : float or None
        Cap on the inflation factor.
    """

    def __init__(
        self,
        max_ratio: float,
        alpha: float = 3.0,
        max_scale: float | None = None,
    ):
        self.max_ratio = max_ratio
        self.alpha = alpha
        self.max_scale = max_scale

    def apply(self, median_asd, max_asd):
        """
        Inflate bins above the ratio threshold logarithmically.

        Parameters and returns as in :meth:`BlackoutPolicy.apply`.
        """
        ratio = max_asd / median_asd

        scale = np.ones_like(median_asd)

        mask = ratio > self.max_ratio
        x = (ratio[mask] - self.max_ratio) / self.max_ratio
        scale[mask] += self.alpha * np.log1p(x)

        if self.max_scale is not None:
            scale = np.minimum(scale, self.max_scale)

        return median_asd * scale, np.where(mask)[0]


class SqrtSoftRatioBlackout(BlackoutPolicy):
    """
    Square-root inflation of bins whose ASD ratio exceeds a threshold.

    The median ASD is multiplied by
    ``1 + alpha * sqrt(ratio / max_ratio - 1)`` at the bins where
    ``ratio = max_asd / median_asd > max_ratio``.

    Parameters
    ----------
    max_ratio : float
        Ratio above which inflation begins.
    alpha : float
        Strength of the inflation.
    max_scale : float or None
        Cap on the inflation factor.
    """

    def __init__(self, max_ratio, alpha=3.0, max_scale=None):
        self.max_ratio = max_ratio
        self.alpha = alpha
        self.max_scale = max_scale

    def apply(self, median_asd, max_asd):
        """
        Inflate bins above the ratio threshold by a square root.

        Parameters and returns as in :meth:`BlackoutPolicy.apply`.
        """
        ratio = max_asd / median_asd
        scale = np.ones_like(median_asd)

        mask = ratio > self.max_ratio
        scale[mask] += self.alpha * np.sqrt(ratio[mask] / self.max_ratio - 1)

        if self.max_scale is not None:
            scale = np.minimum(scale, self.max_scale)

        return median_asd * scale, np.where(mask)[0]


class SmoothSoftRatioBlackout(BlackoutPolicy):
    """
    Power-law inflation as in :class:`SoftRatioBlackout`, smoothed.

    The log of the inflation factor is convolved with a Gaussian kernel
    of width ``smooth_sigma_bins``, which spreads the inflation onto the
    bins next to a line and gives the notch tapered edges.

    Parameters
    ----------
    max_ratio : float
        Ratio above which inflation begins.
    alpha : float
        Strength of the inflation.
    beta : float
        Power-law exponent.
    smooth_sigma_bins : float
        Standard deviation of the Gaussian kernel, in bins; 0 or less
        disables the smoothing.
    max_scale : float or None
        Cap on the inflation factor after smoothing.
    """

    def __init__(
        self,
        max_ratio: float,
        alpha: float = 5.0,
        beta: float = 2.0,
        smooth_sigma_bins: float = 2.0,
        max_scale: float | None = None,
    ):
        self.max_ratio = max_ratio
        self.alpha = alpha
        self.beta = beta
        self.smooth_sigma_bins = smooth_sigma_bins
        self.max_scale = max_scale

    def apply(self, median_asd, max_asd):
        """
        Inflate by a power law, then smooth the log inflation factor.

        Parameters and returns as in :meth:`BlackoutPolicy.apply`; the
        indices are the bins with an inflation factor above 1 + 1e-6.
        """
        ratio = max_asd / median_asd

        scale = np.ones_like(median_asd)
        mask = ratio > self.max_ratio
        scale[mask] += self.alpha * ((ratio[mask] / self.max_ratio) ** self.beta - 1)

        # Kernel cut at 4 sigma; smoothing the log keeps scale >= 1
        sig = float(self.smooth_sigma_bins)
        if sig > 0:
            r = max(1, int(round(4 * sig)))
            k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
            k /= k.sum()
            scale = np.exp(np.convolve(np.log(scale), k, mode="same"))

        if self.max_scale is not None:
            scale = np.minimum(scale, self.max_scale)

        idxs = np.where(scale > 1.0 + 1e-6)[0]
        return median_asd * scale, idxs


class NoBlackout(BlackoutPolicy):
    """Policy that returns the median ASD unchanged."""

    def apply(self, median_asd, max_asd):
        """
        Return the median ASD unchanged and no blackout indices.

        Parameters and returns as in :meth:`BlackoutPolicy.apply`; the
        returned indices are empty.
        """
        return median_asd, np.empty(0, dtype=np.int64)
