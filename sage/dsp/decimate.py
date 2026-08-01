#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : decimate.py
Description     : Anti-alias decimation (half-band Kaiser FIR cascade).

    Standalone power-of-2 decimator for real or complex 1-D signals.  Uses the
    same half-band Kaiser-windowed-sinc FIR design as
    ``sage.dsp.multirate_sampling.MultirateSampler`` (63-tap, Kaiser beta=9,
    ~90 dB stopband) but as a plain *uniform* decimator rather than a
    per-frequency-bin multirate compressor.

    Each pass anti-aliases (half-band lowpass at the new Nyquist) and then
    downsamples by 2; :func:`decimate` cascades ``log2(factor)`` passes.
    Complex input is decimated on its real and imaginary parts (the FIR is
    real, so this is exact).

        h      = halfband_kernel()          # the FIR taps
        y      = decimate(x, factor)        # anti-alias + downsample
        fs, k  = rate_and_factor(B, fs0)    # rate/factor for a bandwidth B

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
    Kaiser-windowed half-band FIR lowpass (cutoff = fs/4), as in MultirateSampler.

    Parameters
    ----------
    ntaps : int
        Number of taps (63 -> ~95 dB stopband with beta=9).
    beta : float
        Kaiser window parameter.
    normalize : {"dc", "l2"}
        ``"dc"``: DC gain 1 (amplitude-preserving; use for *signals*).
        ``"l2"``: unit L2 norm (white-noise-variance-preserving; the
        MultirateSampler convention for whitened data, DC gain != 1).

    Returns
    -------
    numpy.ndarray, shape ``(ntaps,)``
    """
    M = ntaps - 1
    n = np.arange(ntaps)
    h = np.sinc((n - M / 2.0) / 2.0) * np.kaiser(ntaps, beta)   # ideal half-band x window
    if normalize == "dc":
        return h / h.sum()
    if normalize == "l2":
        return h / np.sqrt((h ** 2).sum())
    raise ValueError("normalize must be 'dc' or 'l2'")


def _decimate_by_2(x2d, kernel):
    """One half-band anti-alias pass + downsample by 2.  ``x2d``: (B, L) real."""
    pad = kernel.shape[-1] // 2
    xp = F.pad(x2d.unsqueeze(1), (pad, pad), mode="reflect")    # (B, 1, L+2pad)
    y = F.conv1d(xp, kernel, stride=2)                          # (B, 1, ~L/2)
    return y.squeeze(1)


def decimate(x, factor, *, axis=-1, ntaps=63, beta=9.0, normalize="dc"):
    """
    Anti-alias then downsample ``x`` by ``factor`` (a power of 2) along ``axis``.

    Real or complex; numpy array or torch tensor; arbitrary leading batch
    dimensions.  Returns the same array type as the input.

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
    factor : int
        Decimation factor, a power of 2.
    axis : int
        Axis to decimate (default last).
    ntaps, beta, normalize :
        Forwarded to :func:`halfband_kernel`.

    Returns
    -------
    Decimated array of the same backend as ``x``, with ``axis`` shortened by
    ``~factor``.
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
    Smallest power-of-2 sample rate that satisfies the Nyquist condition for a
    complex baseband of two-sided half-bandwidth ``bandwidth`` (content spans
    ``[-B, +B]``, so the requirement is ``rate >= 2 B``), with a safety
    ``buffer``.  Returns ``(rate, decimation_factor)`` where
    ``decimation_factor = original_fs / rate`` (floored to a power of 2).
    """
    need = 2.0 * bandwidth * buffer
    target = min(2 ** math.ceil(math.log2(need)), original_fs)
    factor = int(original_fs // target)
    factor = 1 << (factor.bit_length() - 1)                   # floor to power of 2
    return original_fs / factor, factor
