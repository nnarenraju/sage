#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Heterodyning of complex series: removing a reference phase, in time or
frequency domain, for NumPy or torch, batched.

Filename        : heterodyne.py
Description     : General signal heterodyning (complex demodulation).

Created on 2026-06-24

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__status__      = inProgress
"""

import numpy as np

try:
    import torch
except ImportError:  # torch is optional for the pure-numpy path
    torch = None


def _is_torch(x):
    """Whether ``x`` is a torch tensor."""
    return torch is not None and isinstance(x, torch.Tensor)


def apply_phase(data, phase):
    """
    Multiply complex ``data`` by ``exp(-i * phase)``.

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        Complex series.
    phase : array-like
        Real phase in radians, broadcastable to ``data``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Complex product, backend of ``data``.

    Raises
    ------
    TypeError
        If ``data`` is not complex.
    """
    if _is_torch(data):
        if not torch.is_complex(data):
            raise TypeError("apply_phase expects complex `data`.")
        phase = torch.as_tensor(phase, device=data.device)
        return data * torch.exp(-1j * phase)

    if not np.iscomplexobj(data):
        raise TypeError("apply_phase expects complex `data`.")
    return data * np.exp(-1j * np.asarray(phase))


def heterodyne(signal, template):
    """
    Demodulate ``signal`` by the phase of ``template``.

    Returns ``signal * exp(-i * angle(template))``: phase only, so the
    template envelope does not weight the result. A real time series
    needs :func:`sage.dsp.helpers.analytic_signal` first.

    Parameters
    ----------
    signal : numpy.ndarray or torch.Tensor
        Complex series: analytic time series or frequency series.
    template : numpy.ndarray or torch.Tensor
        Complex reference, broadcastable to ``signal``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Heterodyned series, backend of ``signal``.
    """
    if _is_torch(signal):
        return signal * torch.exp(-1j * torch.angle(template))
    return signal * np.exp(-1j * np.angle(template))
