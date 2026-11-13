#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inverse spectrum truncation of PSDs in PyTorch, batched or single,
after PyCBC ``inverse_spectrum_truncation``.

Filename        : inverse_spectrum_truncation.py
Description     : Short description of the file

Created on 2026-02-09 12:18:56

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation:

    We take inspiration from the PyCBC function for truncation
    https://pycbc.org/pycbc/latest/html/_modules/pycbc/psd/estimate.html#inverse_spectrum_truncation

"""

import numpy as np
import torch


def inverse_spectrum_truncation_batch(
    psd: torch.Tensor,
    max_filter_len: int,
    low_frequency_cutoff: float = None,
    delta_f: float = 1.0,
    trunc_method: str = "hann",
) -> torch.Tensor:
    """
    Inverse spectrum truncation of a batch of PSDs.

    The inverse ASD on bins ``[kmin, N/2)`` is inverse-FFTed; the
    ``max_filter_len`` samples kept at the ends are optionally tapered,
    the rest zeroed, and the PSD is ``1 / |rFFT|^2`` of the result.

    Parameters
    ----------
    psd : torch.Tensor, shape ``(B, D, F)``
        One-sided PSDs.
    max_filter_len : int
        Length in samples of the time-domain inverse-ASD filter kept.
    low_frequency_cutoff : float or None
        Frequency in Hz below which the inverse ASD is zero; None or 0
        starts at bin 1.
    delta_f : float
        Frequency spacing in Hz.
    trunc_method : str or None
        ``"hann"`` tapers the kept samples with the halves of a
        symmetric Hann window; anything else truncates without a taper.

    Returns
    -------
    torch.Tensor, shape ``(B, D, F)``
        Truncated PSDs.

    Raises
    ------
    ValueError
        If ``max_filter_len`` is not positive or exceeds ``2 (F - 1)``.
    """
    B, D, F = psd.shape
    device = psd.device
    dtype = psd.dtype

    if max_filter_len <= 0:
        raise ValueError("max_filter_len must be positive integer")

    N = 2 * (F - 1)  # full time-domain length for real FFT

    inv_asd = torch.zeros_like(
        psd, dtype=torch.complex64 if dtype == torch.float32 else torch.complex128
    )
    # As PyCBC: the inverse ASD spans [kmin, N/2); DC and Nyquist stay 0
    kmin = 1
    if low_frequency_cutoff:
        kmin = int(low_frequency_cutoff / delta_f)

    inv_asd[..., kmin : N // 2] = 1.0 / torch.sqrt(
        psd[..., kmin : N // 2].to(inv_asd.real.dtype)
    )

    q = torch.fft.irfft(inv_asd, n=N, dim=-1)

    trunc_start = max_filter_len // 2
    trunc_end = N - max_filter_len // 2

    if trunc_end < trunc_start:
        raise ValueError("Invalid max_filter_len too large for PSD length")

    if trunc_method == "hann":
        # PyCBC's window: numpy.hanning (symmetric)
        window = torch.from_numpy(np.hanning(max_filter_len)).to(device=device, dtype=q.dtype)
        q[..., :trunc_start] *= window[-trunc_start:]
        q[..., trunc_end:] *= window[: (N - trunc_end)]

    if trunc_start < trunc_end:
        q[..., trunc_start:trunc_end] = 0.0

    psd_trunc = torch.fft.rfft(q, n=N, dim=-1)
    psd_trunc = psd_trunc.abs() ** 2
    psd_out = 1.0 / psd_trunc

    return psd_out[..., :F]


def inverse_spectrum_truncation_single(
    psd: torch.Tensor,
    max_filter_len: int,
    low_frequency_cutoff: float = None,
    delta_f: float = 1.0,
    trunc_method: str = "hann",
) -> torch.Tensor:
    """
    Inverse spectrum truncation of one PSD, in float64.

    The inverse ASD on bins ``[kmin, N/2)`` is inverse-FFTed; the
    ``max_filter_len`` samples kept at the ends are optionally tapered,
    the rest zeroed, and the PSD is ``1 / |rFFT|^2`` of the result.

    Parameters
    ----------
    psd : torch.Tensor, shape ``(F,)``
        One-sided PSD.
    max_filter_len : int
        Length in samples of the time-domain inverse-ASD filter kept.
    low_frequency_cutoff : float or None
        Frequency in Hz below which the inverse ASD is zero; None or 0
        starts at bin 1.
    delta_f : float
        Frequency spacing in Hz.
    trunc_method : str or None
        ``"hann"`` tapers the kept samples with the halves of a
        symmetric Hann window; anything else truncates without a taper.

    Returns
    -------
    torch.Tensor, shape ``(F,)``
        Truncated PSD, float64, on the device of ``psd``.

    Raises
    ------
    ValueError
        If ``psd`` is not 1D, or ``max_filter_len`` is not positive or
        exceeds ``2 (F - 1)``.
    """
    if not psd.ndim == 1:
        raise ValueError("PSD must be 1D tensor (single PSD)")

    if max_filter_len <= 0:
        raise ValueError("max_filter_len must be positive integer")

    F = psd.shape[0]
    N = 2 * (F - 1)  # full time-domain length

    inv_asd = torch.zeros(F, dtype=torch.complex128, device=psd.device)
    # As PyCBC: the inverse ASD spans [kmin, N/2); DC and Nyquist stay 0
    kmin = 1
    if low_frequency_cutoff:
        kmin = int(low_frequency_cutoff / delta_f)

    inv_asd[kmin : N // 2] = 1.0 / torch.sqrt(psd[kmin : N // 2].to(torch.float64))

    q = torch.fft.irfft(inv_asd, n=N)

    trunc_start = max_filter_len // 2
    trunc_end = N - max_filter_len // 2
    if trunc_end < trunc_start:
        raise ValueError("max_filter_len too large for PSD length")

    if trunc_method == "hann":
        # PyCBC's window: numpy.hanning (symmetric)
        window = torch.from_numpy(np.hanning(max_filter_len)).to(device=q.device, dtype=q.dtype)
        q[:trunc_start] *= window[-trunc_start:]
        q[trunc_end:] *= window[: (N - trunc_end)]

    if trunc_start < trunc_end:
        q[trunc_start:trunc_end] = 0.0

    psd_trunc = torch.fft.rfft(q, n=N)
    psd_trunc = psd_trunc.abs() ** 2
    psd_out = 1.0 / psd_trunc

    return psd_out[:F]
