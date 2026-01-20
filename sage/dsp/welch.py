#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : welch.py
Description     : Short description of the file

Created on 2026-01-20 12:42:08

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

import numpy as np
import scipy.signal as ss


class WelchPSD:
    """
    Welch PSD estimator.

    Thin wrapper around scipy.signal.welch with
    consistent defaults and a stable API.
    """

    def __init__(
        self,
        sample_rate: float,
        nperseg_seconds: float = 4.0,
        average: str = "median",
        detrend: str | None = "constant",
        window: str = "hann",
        scaling: str = "density",
    ):
        """
        Args:
            sample_rate: Sampling rate in Hz
            nperseg_seconds: Segment length in seconds
            average: 'mean' or 'median'
            detrend: Detrending method
            window: Window function
            scaling: 'density' or 'spectrum'
        """
        self.sample_rate = sample_rate
        self.nperseg_seconds = nperseg_seconds
        self.average = average
        self.detrend = detrend
        self.window = window
        self.scaling = scaling

        self.nperseg = int(round(self.nperseg_seconds * self.sample_rate))
        if self.nperseg <= 0:
            raise ValueError("nperseg must be positive")

    def __call__(self, x: np.ndarray):
        """
        Compute PSD.

        Args:
            x: 1D time-series array

        Returns:
            freqs, psd
        """
        if x.ndim != 1:
            raise ValueError("WelchPSD expects a 1D array")

        freqs, pxx = ss.welch(
            x,
            fs=self.sample_rate,
            nperseg=self.nperseg,
            average=self.average,
            detrend=self.detrend,
            window=self.window,
            scaling=self.scaling,
        )
        return freqs, pxx
