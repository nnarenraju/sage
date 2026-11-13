#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analytic signal and instantaneous frequency of NumPy or torch series.

Filename        : helpers.py
Description     : DSP signal helpers.
"""

import numpy as np

try:
    import torch
except ImportError:  # torch is optional for the pure-numpy path
    torch = None


def _is_torch(x):
    """Whether ``x`` is a torch tensor."""
    return torch is not None and isinstance(x, torch.Tensor)


def _is_complex(x):
    """Whether ``x``, a NumPy array or torch tensor, is complex."""
    if _is_torch(x):
        return torch.is_complex(x)
    return np.iscomplexobj(x)


def analytic_signal(x, axis=-1):
    """
    Analytic signal of a real array along ``axis``.

    The FFT construction of :func:`scipy.signal.hilbert`: negative
    frequencies zeroed, positive ones doubled. A complex ``x`` is
    returned unchanged.

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        Real or complex input, any shape.
    axis : int
        Axis of the transform.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Complex analytic signal, shape and backend of ``x``.
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
    Instantaneous frequency of a complex series along the last axis.

    Phase-difference estimator ``angle(z[n+1] conj(z[n])) / (2 pi dt)``;
    no phase unwrapping.

    Parameters
    ----------
    z : numpy.ndarray or torch.Tensor
        Complex series.
    dt : float
        Sample spacing in seconds; for a frequency series, the bin
        spacing, which gives a time per bin instead.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Frequency in Hz (or time per bin), one sample shorter than
        ``z`` along the last axis.
    """
    two_pi = 2.0 * np.pi
    if _is_torch(z):
        prod = z[..., 1:] * torch.conj(z[..., :-1])
        return torch.angle(prod) / (two_pi * dt)
    prod = z[..., 1:] * np.conj(z[..., :-1])
    return np.angle(prod) / (two_pi * dt)
