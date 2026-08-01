#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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

# Packages
import torch


def inverse_spectrum_truncation_batch(
    psd: torch.Tensor,
    max_filter_len: int,
    low_frequency_cutoff: float = None,
    delta_f: float = 1.0,
    trunc_method: str = "hann",
) -> torch.Tensor:
    """
    GPU-native batched inverse spectrum truncation (IST) for PSD smoothing.

    Reproduces the PyCBC ``inverse_spectrum_truncation`` routine in PyTorch
    for batched, device-resident PSDs.  IST time-limits the inverse-ASD
    filter so that it cannot introduce correlations beyond ``max_filter_len``
    samples, preventing the whitening filter from growing unboundedly long.

    Algorithm:

    1. Compute ``1 / sqrt(PSD)`` in the frequency domain (inverse ASD).
    2. IFFT to the time domain and zero out samples beyond ``max_filter_len/2``
       from each edge (optionally with a Hann window at the truncation point).
    3. FFT back and square to recover the truncated PSD.

    Parameters
    ----------
    psd : torch.Tensor, shape ``(B, D, F)``
        Real-valued one-sided PSDs.
    max_filter_len : int
        Maximum whitening filter length in samples.  Bins beyond this radius
        are zeroed in the time-domain inverse ASD.
    low_frequency_cutoff : float or None
        Frequency (Hz) below which the inverse ASD is set to zero.
    delta_f : float
        Frequency bin spacing in Hz (default ``1.0``).
    trunc_method : str or None
        Window applied at the truncation boundary: ``"hann"`` applies a
        half-Hann taper; ``None`` uses a hard rectangular truncation.

    Returns
    -------
    torch.Tensor, shape ``(B, D, F)``
        Truncated PSDs ready for use in whitening.
    """
    B, D, F = psd.shape
    device = psd.device
    dtype = psd.dtype

    if max_filter_len <= 0:
        raise ValueError("max_filter_len must be positive integer")

    N = 2 * (F - 1)  # full time-domain length for real FFT

    # Prepare inverse sqrt PSD in FD
    inv_asd = torch.zeros_like(
        psd, dtype=torch.complex64 if dtype == torch.float32 else torch.complex128
    )
    kmin = 0
    if low_frequency_cutoff is not None:
        kmin = int(low_frequency_cutoff / delta_f)

    # Fill inverse sqrt PSD
    inv_asd[..., kmin:F] = 1.0 / torch.sqrt(psd[..., kmin:F].to(inv_asd.real.dtype))

    # Convert to time domain (complex IFFT)
    q = torch.fft.irfft(inv_asd, n=N, dim=-1)

    # Truncate edges
    trunc_start = max_filter_len // 2
    trunc_end = N - max_filter_len // 2

    if trunc_end < trunc_start:
        raise ValueError("Invalid max_filter_len too large for PSD length")

    # Apply Hann taper if requested
    if trunc_method == "hann":
        window = torch.hann_window(max_filter_len, device=device, dtype=q.dtype)
        # left edge
        q[..., :trunc_start] *= window[-trunc_start:]
        # right edge
        q[..., trunc_end:] *= window[: (N - trunc_end)]

    # Zero out central region if necessary
    if trunc_start < trunc_end:
        q[..., trunc_start:trunc_end] = 0.0

    # Transform back to FD
    psd_trunc = torch.fft.rfft(q, n=N, dim=-1)
    # Take magnitude squared
    psd_trunc = psd_trunc.abs() ** 2
    # Invert
    psd_out = 1.0 / psd_trunc

    # Return only first F bins (same as input)
    return psd_out[..., :F]


def inverse_spectrum_truncation_single(
    psd: torch.Tensor,
    max_filter_len: int,
    low_frequency_cutoff: float = None,
    delta_f: float = 1.0,
    trunc_method: str = "hann",
) -> torch.Tensor:
    """
    CPU version of inverse spectrum truncation for a single PSD (1D tensor).

    Args:
        psd: torch.Tensor, shape (F,), real-valued PSD
        max_filter_len: int, maximum length of the time-domain filter
        low_frequency_cutoff: float or None, zero out PSD below this frequency
        delta_f: float, frequency bin spacing of PSD
        trunc_method: str or None, truncation method ("hann" or None)

    Returns:
        torch.Tensor, shape (F,), truncated PSD
    """
    if not psd.ndim == 1:
        raise ValueError("PSD must be 1D tensor (single PSD)")

    if max_filter_len <= 0:
        raise ValueError("max_filter_len must be positive integer")

    F = psd.shape[0]
    N = 2 * (F - 1)  # full time-domain length

    # Prepare inverse sqrt PSD in FD
    inv_asd = torch.zeros(F, dtype=torch.complex128)  # always use float64 for CPU
    kmin = 0
    if low_frequency_cutoff is not None:
        kmin = int(low_frequency_cutoff / delta_f)

    # Fill inverse sqrt PSD
    inv_asd[kmin:F] = 1.0 / torch.sqrt(psd[kmin:F].to(torch.float64))

    # IFFT to TD
    q = torch.fft.irfft(inv_asd, n=N)

    # Truncate edges
    trunc_start = max_filter_len // 2
    trunc_end = N - max_filter_len // 2
    if trunc_end < trunc_start:
        raise ValueError("max_filter_len too large for PSD length")

    # Apply Hann taper if requested
    if trunc_method == "hann":
        window = torch.hann_window(max_filter_len, dtype=q.dtype)
        # left edge
        q[:trunc_start] *= window[-trunc_start:]
        # right edge
        q[trunc_end:] *= window[: (N - trunc_end)]

    # Zero out center if applicable
    if trunc_start < trunc_end:
        q[trunc_start:trunc_end] = 0.0

    # FFT back to FD
    psd_trunc = torch.fft.rfft(q, n=N)
    psd_trunc = psd_trunc.abs() ** 2
    psd_out = 1.0 / psd_trunc

    # Return only first F bins
    return psd_out[:F]
