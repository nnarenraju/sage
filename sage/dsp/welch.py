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

# Packages
import torch
import numpy as np
import scipy.signal as ss

# LOCAL
from sage.core.conversions import seconds_to_samples


class ScipyWelch:
    """
    Welch PSD estimator.

    Thin wrapper around scipy.signal.welch with
    consistent defaults and a stable API.
    """

    def __init__(
        self,
        sample_rate: float,
        nperseg_in_seconds: float = 4.0,
        average: str = "median",
        detrend: str | None = "constant",
        window: str = "hann",
        scaling: str = "density",
    ):
        """
        Args:
            sample_rate: Sampling rate in Hz
            nperseg_in_seconds: Segment length in seconds
            average: 'mean' or 'median'
            detrend: Detrending method
            window: Window function
            scaling: 'density' or 'spectrum'
        """
        self.sample_rate = sample_rate
        self.nperseg_in_seconds = nperseg_in_seconds
        self.average = average
        self.detrend = detrend
        self.window = window
        self.scaling = scaling

        self.nperseg = seconds_to_samples(self.nperseg_in_seconds, self.sample_rate)
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


class TorchWelch:
    """
    Welch PSD estimator using PyTorch (CPU-friendly).

    Designed for a single time series per call. Returns PSD in frequency domain.
    Refer: https://pycbc.org/pycbc/latest/html/_modules/pycbc/psd/estimate.html

    Attributes:
        delta_t: Sampling interval (seconds)
        seg_len: Segment length (samples)
        seg_stride: Stride between segments (samples)
        window: Window type ('hann') or torch tensor
        avg_method: 'mean', 'median', 'median-mean'
        require_exact_data_fit: if True, enforce exact segment fit
        minimum_segments: minimum number of segments required
    """

    def __init__(
        self,
        delta_t: float = 1.0 / 2048,
        seg_len: int = 4096,
        seg_stride: int = 2048,
        window: str = "hann",
        avg_method: str = "median",
        require_exact_data_fit: bool = False,
        minimum_segments: int = None,
    ):
        self.delta_t = delta_t
        self.seg_len = seg_len
        self.seg_stride = seg_stride
        self.avg_method = avg_method
        self.require_exact_data_fit = require_exact_data_fit
        self.minimum_segments = minimum_segments

        self.delta_f = 1.0 / (self.seg_len * self.delta_t)
        self.freqs = (
            torch.arange(self.seg_len // 2 + 1, dtype=torch.float64) * self.delta_f
        )

        # Window
        if isinstance(window, str):
            if window.lower() == "hann":
                self.window = torch.hann_window(self.seg_len)
            else:
                raise ValueError(f"Unknown window type {window}")
        else:
            if len(window) != seg_len:
                raise ValueError("Window length does not match seg_len")
            self.window = window

    @torch.no_grad()
    def __call__(self, timeseries: torch.Tensor) -> torch.Tensor:
        """
        Compute PSD for a single 1D timeseries.

        Args:
            timeseries: torch.Tensor of shape (N,), float32 or float64

        Returns:
            psd: torch.Tensor of shape (seg_len//2 + 1,)
        """
        if timeseries.ndim != 1:
            raise ValueError("Timeseries must be 1D")

        N = timeseries.shape[0]
        timeseries = timeseries.to(dtype=torch.float64)

        # Number of segments
        num_segments = (N - self.seg_len) // self.seg_stride + 1
        if self.minimum_segments is not None and num_segments < self.minimum_segments:
            raise ValueError("Not enough segments for PSD estimation")

        # Trim to exact fit if allowed
        if not self.require_exact_data_fit:
            data_len = (num_segments - 1) * self.seg_stride + self.seg_len
            if data_len < N:
                diff = N - data_len
                start = diff // 2
                end = N - (diff - diff // 2)
                timeseries = timeseries[start:end]
                N = timeseries.shape[0]

        if N != (num_segments - 1) * self.seg_stride + self.seg_len:
            raise ValueError("Inconsistent segmentation parameters")

        delta_f = 1.0 / (self.seg_len * self.delta_t)
        segment_psds = []

        # Normalizes power of window
        fs = 1.0 / self.delta_t
        # window power normalization
        U = (self.window**2).sum()

        # Compute PSD for each segment
        for i in range(num_segments):
            start = i * self.seg_stride
            segment = timeseries[start : start + self.seg_len]
            # Constant detrending similar to scipy welch
            segment = segment - segment.mean()
            segment = segment * self.window
            fft_seg = torch.fft.rfft(segment)
            # Correct scaling factors
            seg_psd = fft_seg.abs() ** 2 / (fs * U)
            seg_psd[1:-1] *= 2  # one-sided
            segment_psds.append(seg_psd)

        segment_psds = torch.stack(segment_psds, dim=0)

        # Average segments
        if self.avg_method == "mean":
            psd = torch.mean(segment_psds, dim=0)
        elif self.avg_method == "median":
            psd = torch.median(segment_psds, dim=0).values
            psd /= self._median_bias(num_segments)
        elif self.avg_method == "median-mean":
            odd = segment_psds[::2]
            even = segment_psds[1::2]
            odd_med = torch.median(odd, dim=0).values / self._median_bias(len(odd))
            even_med = torch.median(even, dim=0).values / self._median_bias(len(even))
            psd = 0.5 * (odd_med + even_med)
        else:
            raise ValueError(f"Unknown avg_method {self.avg_method}")

        return psd

    @staticmethod
    def _median_bias(n: int) -> float:
        """Median bias correction for PSD estimation"""
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if n >= 1000:
            return torch.log(torch.tensor(2.0)).item()

        ans = 1.0
        for i in range(1, (n - 1) // 2 + 1):
            ans += 1.0 / (2 * i + 1) - 1.0 / (2 * i)

        return ans
