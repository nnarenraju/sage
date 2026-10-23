#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : helpers.py
Description     : DSP signal helpers.

    Utilities for building and inspecting complex baseband signals — used by the
    heterodyne demos/tests and any caller that needs an analytic signal or an
    instantaneous-frequency estimate.  Kept in a separate module from
    ``sage.dsp.heterodyne`` so the heterodyne hot path stays minimal
    (complex-in / complex-out, no Hilbert transforms).

        analytic_signal(x)          real -> analytic signal (Hilbert via FFT)
        instantaneous_frequency(z)  instantaneous frequency of a complex series
"""

import numpy as np

try:
    import torch
except ImportError:  # torch is optional for the pure-numpy path
    torch = None


def _is_torch(x):
    return torch is not None and isinstance(x, torch.Tensor)


def _is_complex(x):
    if _is_torch(x):
        return torch.is_complex(x)
    return np.iscomplexobj(x)


def analytic_signal(x, axis=-1):
    """
    Return the analytic signal of a real array along ``axis``.

    Uses the standard FFT construction (identical to
    :func:`scipy.signal.hilbert`): zero the negative-frequency half, double the
    positive-frequency half.  ``z = x + i * Hilbert(x)`` satisfies
    ``z.real == x`` and has only non-negative frequency content, so multiplying
    by ``exp(-i * phase)`` shifts it cleanly to baseband without generating
    image (sum-frequency) components.  A complex ``x`` is returned unchanged.

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        Real (or complex) input, any shape.
    axis : int
        Axis along which to compute the analytic signal (default last).

    Returns
    -------
    Complex array of the same shape and backend as ``x``.
    """
    if _is_complex(x):
        return x

    n = x.shape[axis]

    if _is_torch(x):
        X = torch.fft.fft(x, dim=axis)
        h = torch.zeros(n, dtype=X.real.dtype, device=x.device)
        if n % 2 == 0:
            h[0] = 1.0
            h[n // 2] = 1.0
            h[1 : n // 2] = 2.0
        else:
            h[0] = 1.0
            h[1 : (n + 1) // 2] = 2.0
        shape = [1] * x.ndim
        shape[axis] = n
        return torch.fft.ifft(X * h.reshape(shape), dim=axis)

    X = np.fft.fft(x, axis=axis)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    shape = [1] * x.ndim
    shape[axis] = n
    return np.fft.ifft(X * h.reshape(shape), axis=axis)


def instantaneous_frequency(z, dt=1.0):
    """
    Instantaneous frequency (Hz) of a complex series along the last axis.

    Uses the phase-difference estimator
    ``f[n] = angle(z[n+1] * conj(z[n])) / (2*pi*dt)`` — no phase unwrapping,
    robust for slowly varying baseband signals.  Returns an array one sample
    shorter than ``z`` along the last axis.

    Parameters
    ----------
    z : complex numpy.ndarray or torch.Tensor
    dt : float
        Sample spacing (seconds).  For frequency-domain input, pass the bin
        spacing ``df`` to get a "chirp time" per bin instead.
    """
    two_pi = 2.0 * np.pi
    if _is_torch(z):
        prod = z[..., 1:] * torch.conj(z[..., :-1])
        return torch.angle(prod) / (two_pi * dt)
    prod = z[..., 1:] * np.conj(z[..., :-1])
    return np.angle(prod) / (two_pi * dt)
