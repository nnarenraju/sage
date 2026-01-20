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
    def apply(self, median_psd, psds):
        return median_psd, None


class HardRatioBlackout(BlackoutPolicy):
    def __init__(self, max_ratio):
        self.max_ratio = max_ratio

    def apply(self, median_psd, psds):
        max_psd = np.max(psds, axis=0)
        ratio = max_psd / median_psd
        idxs = np.where(ratio > self.max_ratio)[0]

        psd = median_psd.copy()
        psd[idxs] = 1e12

        frac = len(idxs) / len(ratio)
        print(f"[{self.detector}] " f"Blacked out {frac*100:.2f}% of frequency bins")

        return psd, idxs


class SoftRatioBlackout(BlackoutPolicy):
    def __init__(
        self,
        max_ratio: float,
        alpha: float = 5.0,
        beta: float = 2.0,
        max_scale: float | None = None,
    ):
        """
        Args:
            max_ratio: threshold ratio to start suppressing
            alpha: overall strength of suppression
            beta: curvature (beta > 1 = sharper)
            max_scale: optional cap on PSD inflation
        """
        self.max_ratio = max_ratio
        self.alpha = alpha
        self.beta = beta
        self.max_scale = max_scale

    def apply(self, median_psd, psds):
        max_psd = np.max(psds, axis=0)
        ratio = max_psd / median_psd

        scale = np.ones_like(median_psd)

        mask = ratio > self.max_ratio
        scale[mask] += self.alpha * ((ratio[mask] / self.max_ratio) ** self.beta - 1)

        if self.max_scale is not None:
            scale = np.minimum(scale, self.max_scale)

        return median_psd * scale, np.where(mask)[0]


class GaussianSoftNotchBlackout(BlackoutPolicy):
    def __init__(self, freqs, centers, widths, depth=10.0):
        self.freqs = freqs
        self.centers = centers
        self.widths = widths
        self.depth = depth

    def apply(self, median_psd, psds):
        scale = np.ones_like(median_psd)

        for f0, w in zip(self.centers, self.widths):
            scale += self.depth * np.exp(-0.5 * ((self.freqs - f0) / w) ** 2)

        return median_psd * scale, np.empty(0, dtype=np.int64)


class LogSoftRatioBlackout(BlackoutPolicy):
    def __init__(
        self,
        max_ratio: float,
        alpha: float = 3.0,
        max_scale: float | None = None,
    ):
        """
        Args:
            max_ratio: threshold to start suppressing
            alpha: overall strength (log grows slowly!)
            max_scale: optional hard cap on scaling
        """
        self.max_ratio = max_ratio
        self.alpha = alpha
        self.max_scale = max_scale

    def apply(self, median_psd, psds):
        max_psd = np.max(psds, axis=0)
        ratio = max_psd / median_psd

        scale = np.ones_like(median_psd)

        mask = ratio > self.max_ratio
        x = (ratio[mask] - self.max_ratio) / self.max_ratio
        scale[mask] += self.alpha * np.log1p(x)

        if self.max_scale is not None:
            scale = np.minimum(scale, self.max_scale)

        return median_psd * scale, np.where(mask)[0]


class SqrtSoftRatioBlackout(BlackoutPolicy):
    def __init__(self, max_ratio, alpha=3.0, max_scale=None):
        self.max_ratio = max_ratio
        self.alpha = alpha
        self.max_scale = max_scale

    def apply(self, median_psd, psds):
        ratio = np.max(psds, axis=0) / median_psd
        scale = np.ones_like(median_psd)

        mask = ratio > self.max_ratio
        scale[mask] += self.alpha * np.sqrt(ratio[mask] / self.max_ratio - 1)

        if self.max_scale is not None:
            scale = np.minimum(scale, self.max_scale)

        return median_psd * scale, np.where(mask)[0]


class NoBlackout(BlackoutPolicy):
    def apply(self, median_psd, psds):
        return median_psd, np.empty(0, dtype=np.int64)
