#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Resampling and high-pass filtering of strain: PyCBC downsampling and a
zero-phase Butterworth high-pass.

Filename        : filters.py
Description     : Short description of the file

Created on 2025-12-12 15:44:50

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
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

from scipy.signal import (
    butter,
    sosfiltfilt,
    sosfilt,
    firwin,
    fftconvolve,
    sosfreqz,
)

from pycbc.filter import highpass as pycbc_highpass
from pycbc.types import TimeSeries as TS
from pycbc.filter.resample import resample_to_delta_t

from sage.core.utils import ensure_1d
from sage.dsp.utils import trim_edges
from sage.core.logger import get_logger

logger = get_logger(__name__)


def pycbc_downsample(
    strain,
    old_sample_rate,
    new_sample_rate=2048.0,
    trim=0.2,
    noise_low_freq_cutoff=15.0,
):
    """
    Resample and high-pass a strain time series with PyCBC.

    PyCBC ``resample_to_delta_t`` (default Butterworth method), then
    PyCBC ``highpass``, a cast to float32 and :func:`trim_edges`.

    Parameters
    ----------
    strain : array-like, shape ``(N,)``
        Time series at ``old_sample_rate``.
    old_sample_rate : float
        Input sample rate in Hz.
    new_sample_rate : float
        Output sample rate in Hz.
    trim : float
        Seconds removed from each end.
    noise_low_freq_cutoff : float
        High-pass frequency in Hz.

    Returns
    -------
    numpy.ndarray
        Resampled, high-passed, trimmed strain, float32.

    Raises
    ------
    ValueError
        If the trim is zero samples or at least half the data length.
    """

    pycbc_strain = TS(strain, delta_t=1.0 / old_sample_rate)
    res = resample_to_delta_t(pycbc_strain, delta_t=1.0 / new_sample_rate)
    ret = pycbc_highpass(res, noise_low_freq_cutoff).numpy()
    ret = ret.astype(np.float32, copy=False)

    # Remove the samples corrupted by the filters
    ret = trim_edges(ret, new_sample_rate, trim)
    return ret


def highpass(
    x,
    fs,
    cutoff=15.0,
    order=4,
    padlen_seconds=2.0,
    pad_type="reflect",
    fallback_to_sos=False,
):
    """
    Zero-phase Butterworth high-pass of a padded time series.

    Padded on each side by ``padlen_seconds`` (at least 3 samples;
    ``n // 2 - 1`` if that reaches the length ``n``), filtered forward
    and backward, then unpadded.

    Parameters
    ----------
    x : array-like
        1D time series.
    fs : float
        Sampling rate in Hz.
    cutoff : float
        Cutoff frequency in Hz.
    order : int
        Butterworth order.
    padlen_seconds : float
        Padding on each side in seconds.
    pad_type : str
        ``"reflect"``, ``"symmetric"``, ``"odd"`` or ``"constant"``
        (zeros).
    fallback_to_sos : bool
        If ``sosfiltfilt`` raises, filter causally with ``sosfilt``
        instead of re-raising.

    Returns
    -------
    numpy.ndarray
        Filtered series, length of ``x``; an empty ``x`` is returned
        as is.

    Raises
    ------
    ValueError
        If ``cutoff`` is not inside ``(0, fs/2)`` or ``pad_type`` is
        unknown.
    """

    x = ensure_1d(x)
    n = x.shape[0]
    if n == 0:
        return x

    nyq = 0.5 * fs
    if cutoff <= 0 or cutoff >= nyq:
        logger.error("cutoff must be between 0 and Nyquist (fs/2)")
        raise ValueError("cutoff must be between 0 and Nyquist (fs/2)")

    sos = butter(order, cutoff / nyq, btype="highpass", output="sos")

    padlen = int(max(3, padlen_seconds * fs))
    if padlen >= n:
        padlen = max(0, n // 2 - 1)

    if padlen > 0:
        if pad_type == "reflect":
            xp = np.pad(x, padlen, mode="reflect")
        elif pad_type == "symmetric":
            xp = np.pad(x, padlen, mode="symmetric")
        elif pad_type == "odd":
            xp = np.pad(x, padlen, mode="odd")
        elif pad_type == "constant":
            xp = np.pad(x, padlen, mode="constant", constant_values=0.0)
        else:
            logger.error("unsupported pad_type")
            raise ValueError("unsupported pad_type")
    else:
        xp = x

    try:
        y = sosfiltfilt(sos, xp)
    except Exception as exc:
        # Causal sosfilt if requested, otherwise re-raise
        logger.warning("sosfiltfilt failed; falling back to sosfilt if enabled")
        if fallback_to_sos:
            y = sosfilt(sos, xp)
        else:
            logger.warning("sosfilt fallback failed")
            logger.info("Set fallback_to_sos to True to enable fallback")
            raise

    if padlen > 0:
        y = y[padlen : padlen + n]

    return y


"""
class BandPass(TransformWrapper):
    def __init__(self, always_apply=True, lower=16, upper=512, fs=2048, order=5):
        super().__init__(always_apply)
        self.lower = lower
        self.upper = upper
        self.fs = fs
        self.order = order

    def butter_bandpass(self):
        nyq = 0.5 * self.fs
        low = self.lower / nyq
        high = self.upper / nyq
        sos = butter(
            self.order, [low, high], analog=False, btype="bandpass", output="sos"
        )
        return sos

    def butter_bandpass_filter(self, data):
        sos = self.butter_bandpass()
        filtered_data = sosfiltfilt(sos, data)
        return filtered_data

    def apply(self, y: np.ndarray, special: dict):
        return self.butter_bandpass_filter(y)


class HighPass(TransformWrapper):
    def __init__(self, always_apply=True, lower=16, fs=2048, order=5):
        super().__init__(always_apply)
        self.lower = lower
        self.fs = fs
        self.order = order

    def butter_highpass(self):
        nyq = 0.5 * self.fs
        low = self.lower / nyq
        sos = butter(self.order, low, analog=False, btype="highpass", output="sos")
        return sos

    def butter_highpass_filter(self, data):
        sos = self.butter_highpass()
        filtered_data = sosfiltfilt(sos, data)
        return filtered_data

    def apply(self, y: np.ndarray, special: dict):
        # Parallelise HighPass filter
        return self.butter_highpass_filter(y)

"""
