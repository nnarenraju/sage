#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : blackout.py
Description     : Short description of the file

Created on 2026-01-20 12:47:25

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


class BlackoutPolicy:
    """
    Abstract base class for PSD blackout / glitch-suppression policies.

    A blackout policy inflates (or hard-sets) selected frequency bins of the
    median PSD so that those bins contribute negligibly to the matched-filter
    SNR.  This is used to suppress persistent spectral lines or short-duration
    glitches whose ratio to the median PSD exceeds an acceptable threshold.

    All subclasses must implement :meth:`apply`.

    Parameters
    ----------
    None — subclasses define their own constructor arguments.

    Returns (from :meth:`apply`)
    ----------------------------
    psd : numpy.ndarray, shape ``(F,)``
        Modified (inflated) PSD array.
    idxs : numpy.ndarray of int or None
        Indices of frequency bins that were modified.
    """

    def apply(self, median_psd, max_psd):
        """
        Apply the blackout policy to the median PSD.

        Parameters
        ----------
        median_psd : numpy.ndarray, shape ``(F,)``
            Median PSD computed across all available noise segments.
        max_psd : numpy.ndarray, shape ``(F,)``
            Per-bin maximum PSD across all sampled segments, used to detect
            outlier bins. (Every policy needs only this reduction, not the full
            per-segment stack, so it can be accumulated incrementally.)

        Returns
        -------
        psd : numpy.ndarray, shape ``(F,)``
            Modified PSD (inflated at blackout bins; unchanged elsewhere).
        idxs : numpy.ndarray of int or None
            Indices of frequency bins that were modified, or ``None``.
        """
        return median_psd, None


class HardRatioBlackout(BlackoutPolicy):
    """
    Hard zeroing of frequency bins where ``max_psd / median_psd > max_ratio``.

    Any bin whose worst-case PSD (across all segments) exceeds the median by
    more than ``max_ratio`` is set to ``1e12``, effectively removing it from
    matched-filter calculations.  This is the most aggressive strategy and
    should be used when glitch contamination is severe and well-localised in
    frequency.

    Parameters
    ----------
    max_ratio : float
        Ratio threshold above which a bin is hard-blacked-out.
    """

    def __init__(self, max_ratio):
        self.max_ratio = max_ratio

    def apply(self, median_psd, max_psd):
        """Hard-zero bins where max/median exceeds :attr:`max_ratio` (set to 1e12)."""
        ratio = max_psd / median_psd
        idxs = np.where(ratio > self.max_ratio)[0]

        psd = median_psd.copy()
        psd[idxs] = 1e12

        # We can log this if necessary
        blacked_out_frac = len(idxs) / len(ratio)

        return psd, idxs


class SoftRatioBlackout(BlackoutPolicy):
    """
    Power-law soft suppression of frequency bins with elevated PSD ratios.

    Rather than hard-zeroing, this policy continuously inflates the median PSD
    for bins where ``max_psd / median_psd > max_ratio`` using a power-law
    scale:  ``scale = 1 + alpha * ((ratio / max_ratio)^beta - 1)``.  Bins
    below the threshold are unmodified.  The power-law shape (``beta > 1``)
    makes suppression sharper near the threshold.

    Parameters
    ----------
    max_ratio : float
        Ratio threshold above which suppression begins.
    alpha : float
        Overall strength of the suppression (default ``5.0``).
    beta : float
        Curvature exponent; values > 1 give a sharper onset (default ``2.0``).
    max_scale : float or None
        Optional hard cap on the inflation scale factor (default ``None``).
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

    def apply(self, median_psd, max_psd):
        """Apply power-law soft suppression to elevated PSD bins."""
        ratio = max_psd / median_psd

        scale = np.ones_like(median_psd)

        mask = ratio > self.max_ratio
        scale[mask] += self.alpha * ((ratio[mask] / self.max_ratio) ** self.beta - 1)

        if self.max_scale is not None:
            scale = np.minimum(scale, self.max_scale)

        return median_psd * scale, np.where(mask)[0]


class GaussianSoftNotchBlackout(BlackoutPolicy):
    """
    Soft notch filter: inflates the PSD at known spectral lines via Gaussians.

    Adds a sum of Gaussian bumps centred at ``centers`` with widths ``widths``
    to the median PSD scale factor.  This is useful for suppressing well-known
    spectral lines (e.g. 60 Hz power-grid harmonics, violin modes) without
    hard-removing nearby clean bins.

    Parameters
    ----------
    freqs : array-like, shape ``(F,)``
        Frequency array corresponding to the PSD bins (Hz).
    centers : list[float]
        Centre frequencies of the notch Gaussians (Hz).
    widths : list[float]
        Standard deviations of the notch Gaussians (Hz).
    depth : float
        Peak inflation factor for each Gaussian bump (default ``10.0``).
    """

    def __init__(self, freqs, centers, widths, depth=10.0):
        self.freqs = freqs
        self.centers = centers
        self.widths = widths
        self.depth = depth

    def apply(self, median_psd, max_psd):
        """Inflate the PSD at known spectral lines via Gaussian notch bumps."""
        scale = np.ones_like(median_psd)

        for f0, w in zip(self.centers, self.widths):
            scale += self.depth * np.exp(-0.5 * ((self.freqs - f0) / w) ** 2)

        return median_psd * scale, np.empty(0, dtype=np.int64)


class LogSoftRatioBlackout(BlackoutPolicy):
    """
    Logarithmic soft suppression of bins with elevated PSD ratios.

    Inflates bins where ``max_psd / median_psd > max_ratio`` using a
    logarithmic scale: ``scale = 1 + alpha * log1p((ratio - max_ratio) /
    max_ratio)``.  The log growth is slower than the power-law variant
    (:class:`SoftRatioBlackout`), making this a gentler alternative suitable
    when the ratio can be very large but hard suppression would hurt sensitivity
    too much.

    Parameters
    ----------
    max_ratio : float
        Ratio threshold above which suppression begins.
    alpha : float
        Overall suppression strength (default ``3.0``).
    max_scale : float or None
        Optional hard cap on the inflation scale factor (default ``None``).
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

    def apply(self, median_psd, max_psd):
        """Apply logarithmic soft suppression to elevated PSD bins."""
        ratio = max_psd / median_psd

        scale = np.ones_like(median_psd)

        mask = ratio > self.max_ratio
        x = (ratio[mask] - self.max_ratio) / self.max_ratio
        scale[mask] += self.alpha * np.log1p(x)

        if self.max_scale is not None:
            scale = np.minimum(scale, self.max_scale)

        return median_psd * scale, np.where(mask)[0]


class SqrtSoftRatioBlackout(BlackoutPolicy):
    """
    Square-root soft suppression of bins with elevated PSD ratios.

    Inflates bins where ``max_psd / median_psd > max_ratio`` using a
    square-root scale: ``scale = 1 + alpha * sqrt(ratio / max_ratio - 1)``.
    The sqrt growth is intermediate in aggressiveness between the log
    (:class:`LogSoftRatioBlackout`) and power-law (:class:`SoftRatioBlackout`)
    variants.

    Parameters
    ----------
    max_ratio : float
        Ratio threshold above which suppression begins.
    alpha : float
        Overall suppression strength (default ``3.0``).
    max_scale : float or None
        Optional hard cap on the inflation scale factor (default ``None``).
    """

    def __init__(self, max_ratio, alpha=3.0, max_scale=None):
        self.max_ratio = max_ratio
        self.alpha = alpha
        self.max_scale = max_scale

    def apply(self, median_psd, max_psd):
        """Apply square-root soft suppression to elevated PSD bins."""
        ratio = max_psd / median_psd
        scale = np.ones_like(median_psd)

        mask = ratio > self.max_ratio
        scale[mask] += self.alpha * np.sqrt(ratio[mask] / self.max_ratio - 1)

        if self.max_scale is not None:
            scale = np.minimum(scale, self.max_scale)

        return median_psd * scale, np.where(mask)[0]


class SmoothSoftRatioBlackout(BlackoutPolicy):
    """
    Power-law soft suppression with a Gaussian-tapered (smoothed) profile.

    Identical core to :class:`SoftRatioBlackout` -- inflate bins where
    ``max_psd / median_psd > max_ratio`` by ``1 + alpha*((ratio/max_ratio)**beta - 1)``
    -- but the resulting *scale* profile is Gaussian-smoothed in log-space before it
    is applied.  Two benefits, both verified empirically on O3b HL:

    * **Skirt coverage.**  The raw ratio threshold catches only the line *core*;
      the adjacent skirt bins (ratio just below ``max_ratio``) leak through and
      dominate the whitened-noise residual.  Smoothing spreads the inflation onto
      the skirt, cutting the worst-case whitened-line residual by ~40x vs the hard
      threshold (spikeMEAN 2.2e3 -> ~56 on H1) for ~1% more bins touched.
    * **No ringing.**  A smooth (tapered) notch has no sharp spectral edges, so its
      whitening kernel does not ring in the time domain (Gibbs) -- measured tail
      energy 0.00%, the lowest of all policies.  This is the classic tapered-notch
      SP design; a hard/zeroed notch is the worst case for TD ringing.

    Parameters
    ----------
    max_ratio : float
        Ratio threshold above which suppression begins.
    alpha : float
        Overall suppression strength (default ``5.0``).
    beta : float
        Power-law curvature; > 1 sharpens the onset before smoothing (default ``2.0``).
    smooth_sigma_bins : float
        Gaussian smoothing width, in frequency bins, applied to the log-scale
        profile (default ``2.0``).  Larger = wider skirt coverage + gentler edges.
    max_scale : float or None
        Optional hard cap on the (post-smoothing) inflation factor (default ``None``).
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

    def apply(self, median_psd, max_psd):
        """Power-law soft suppression, then Gaussian-smooth the scale in log-space."""
        ratio = max_psd / median_psd

        scale = np.ones_like(median_psd)
        mask = ratio > self.max_ratio
        scale[mask] += self.alpha * ((ratio[mask] / self.max_ratio) ** self.beta - 1)

        # Gaussian-taper the scale in log-space: covers the line skirt and removes
        # sharp edges (no TD ringing). Smoothing in log keeps scale >= 1 everywhere.
        sig = float(self.smooth_sigma_bins)
        if sig > 0:
            r = max(1, int(round(4 * sig)))
            k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
            k /= k.sum()
            scale = np.exp(np.convolve(np.log(scale), k, mode="same"))

        if self.max_scale is not None:
            scale = np.minimum(scale, self.max_scale)

        idxs = np.where(scale > 1.0 + 1e-6)[0]
        return median_psd * scale, idxs


class NoBlackout(BlackoutPolicy):
    """
    Pass-through policy that leaves the median PSD completely unmodified.

    Returns the median PSD unchanged with an empty blackout index array.
    Used as a no-op default so that code that calls :meth:`apply` always
    gets a consistent interface regardless of whether blackout is enabled.
    """

    def apply(self, median_psd, max_psd):
        """Return the median PSD unmodified with an empty blackout index array."""
        return median_psd, np.empty(0, dtype=np.int64)
