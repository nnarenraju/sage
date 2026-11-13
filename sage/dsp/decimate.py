#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Uniform power-of-2 decimation by a cascade of half-band Kaiser FIR
passes, the filter design of ``MultirateSampler``.

Filename        : decimate.py
Description     : Anti-alias decimation (half-band Kaiser FIR cascade).

Created on 2026-06-24

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__status__      = inProgress
"""

import math
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # torch is required for the conv, optional at import time
    torch = None


def halfband_kernel(ntaps=63, beta=9.0, normalize="dc"):
    """
    Kaiser-windowed half-band FIR low-pass, cutoff ``fs/4``.

    Parameters
    ----------
    ntaps : int
        Number of taps.
    beta : float
        Kaiser window parameter.
    normalize : {"dc", "l2"}
        ``"dc"``: unit DC gain, preserving amplitude. ``"l2"``: unit L2
        norm, preserving white-noise variance (as ``MultirateSampler``).

    Returns
    -------
    numpy.ndarray, shape ``(ntaps,)``
        Filter taps.

    Raises
    ------
    ValueError
        If ``normalize`` is not ``"dc"`` or ``"l2"``.
    """
    M = ntaps - 1
    n = np.arange(ntaps)
    # Ideal half-band response times the Kaiser window
    h = np.sinc((n - M / 2.0) / 2.0) * np.kaiser(ntaps, beta)
    if normalize == "dc":
        return h / h.sum()
    if normalize == "l2":
        return h / np.sqrt((h ** 2).sum())
    raise ValueError("normalize must be 'dc' or 'l2'")


def _decimate_by_2(x2d, kernel):
    """
    One half-band low-pass pass and downsample by 2.

    Parameters
    ----------
    x2d : torch.Tensor, shape ``(B, L)``
        Real series, reflect-padded by ``ntaps // 2`` before filtering.
    kernel : torch.Tensor, shape ``(1, 1, ntaps)``
        Filter taps.

    Returns
    -------
    torch.Tensor, shape ``(B, ceil(L/2))``
        Decimated series, for an odd ``ntaps``.
    """
    pad = kernel.shape[-1] // 2
    xp = F.pad(x2d.unsqueeze(1), (pad, pad), mode="reflect")    # (B, 1, L+2pad)
    y = F.conv1d(xp, kernel, stride=2)                          # (B, 1, ~L/2)
    return y.squeeze(1)


def decimate(x, factor, *, axis=-1, ntaps=63, beta=9.0, normalize="dc"):
    """
    Anti-alias and downsample ``x`` by ``factor`` along ``axis``.

    ``log2(factor)`` half-band passes; a complex ``x`` is decimated on
    its real and imaginary parts.

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        Real or complex series, any leading dimensions.
    factor : int
        Decimation factor, a power of 2.
    axis : int
        Axis to decimate.
    ntaps, beta, normalize
        Passed to :func:`halfband_kernel`.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Decimated series, backend of ``x``, ``axis`` halved (rounded
        up) per pass.

    Raises
    ------
    ValueError
        If ``factor`` is not a power of 2.
    ImportError
        If torch is not installed.
    """
    if factor < 1 or (factor & (factor - 1)) != 0:
        raise ValueError("factor must be a power of 2")
    if torch is None:
        raise ImportError("sage.dsp.decimate requires torch")
    n_pass = int(round(math.log2(factor)))
    if n_pass == 0:
        return x

    was_numpy = not isinstance(x, torch.Tensor)
    xt = torch.movedim(torch.as_tensor(x), axis, -1)           # decimate on last axis
    lead = xt.shape[:-1]
    xt = xt.reshape(-1, xt.shape[-1])                          # (B, L)

    is_complex = torch.is_complex(xt)
    if is_complex:
        real_dtype = xt.real.dtype
        xt = torch.cat([xt.real, xt.imag], dim=0)              # (2B, L) real
    else:
        real_dtype = xt.dtype

    kernel = torch.as_tensor(halfband_kernel(ntaps, beta, normalize),
                             dtype=real_dtype, device=xt.device).view(1, 1, -1)
    for _ in range(n_pass):
        xt = _decimate_by_2(xt, kernel)

    if is_complex:
        b = xt.shape[0] // 2
        xt = torch.complex(xt[:b], xt[b:])
    xt = torch.movedim(xt.reshape(*lead, -1), -1, axis)
    return xt.numpy() if was_numpy else xt


def rate_and_factor(bandwidth, original_fs, buffer=1.5):
    """
    Sample rate and decimation factor for a complex baseband.

    The target is the power of 2 at or above ``2 * bandwidth * buffer``,
    capped at ``original_fs``; the factor ``original_fs // target`` is
    floored to a power of 2.

    Parameters
    ----------
    bandwidth : float
        Half-bandwidth in Hz: the content spans ``[-bandwidth,
        bandwidth]``.
    original_fs : float
        Input sample rate in Hz.
    buffer : float
        Safety factor on the Nyquist rate.

    Returns
    -------
    rate : float
        ``original_fs / factor`` in Hz.
    factor : int
        Decimation factor, a power of 2.
    """
    need = 2.0 * bandwidth * buffer
    target = min(2 ** math.ceil(math.log2(need)), original_fs)
    factor = int(original_fs // target)
    factor = 1 << (factor.bit_length() - 1)                   # floor to power of 2
    return original_fs / factor, factor
