#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : filters.py
Description     : Short description of the file

Created on 2025-12-12 15:44:50

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
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
from scipy.signal import (
    butter,
    sosfiltfilt,
    sosfilt,
    firwin,
    fftconvolve,
    sosfreqz,
)

# LOCAL
from sage.core.utils import ensure_1d


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
    Robust zero-phase highpass using Butterworth SOS + filtfilt-like forward-backward.
    NOTE: Padding reduces edge effects

    Args:
        x (array): 1D time series (float).
        fs (float): sampling rate (Hz).
        cutoff (float): cutoff frequency (Hz).
        order (int): nominal Butterworth order (sos built from order).
        padlen_seconds (float): pad length in seconds (mirror padding)
        pad_type: 'reflect' or 'odd' or 'constant' (prefer 'reflect').
        fallback_to_sos: if True and sosfiltfilt fails, do sosfilt (causal).

    Returns:
        y (np.ndarray): filtered 1D array same length as input.

    """

    x = ensure_1d(x)
    n = x.shape[0]
    if n == 0:
        return x

    nyq = 0.5 * fs
    if cutoff <= 0 or cutoff >= nyq:
        logger.error("cutoff must be between 0 and Nyquist (fs/2)")
        raise ValueError("cutoff must be between 0 and Nyquist (fs/2)")

    # Build SOS
    sos = butter(order, cutoff / nyq, btype="highpass", output="sos")

    # compute pad length in samples (must be >= some multiple of filter transient)
    padlen = int(max(3, padlen_seconds * fs))
    # ensure padlen < n
    if padlen >= n:
        padlen = max(0, n // 2 - 1)

    # Apply padding
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

    # Filter with zero-phase sosfiltfilt
    try:
        y = sosfiltfilt(sos, xp)
    except Exception as exc:
        # fallback: try sosfilt (causal) if requested, otherwise re-raise
        logger.warning("sosfiltfilt failed; falling back to sosfilt if enabled")
        if fallback_to_sos:
            y = sosfilt(sos, xp)
        else:
            logger.warning("sosfilt fallback failed")
            logger.info("Set fallback_to_sos to True to enable fallback")
            raise

    # remove padding
    if padlen > 0:
        y = y[padlen : padlen + n]

    return y