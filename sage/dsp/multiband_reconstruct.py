"""
LAL-equivalent multiband reconstruction (order 1), in PyTorch.

Reconstructs the full-resolution FD waveform ``h(f) = A(f) · exp(i·φ(f))`` from
its samples on a coarse (multiband) frequency grid, replicating the reconstruction
in ``XLALSimIMRPhenomXHMMultiBandOneMode`` with the default
``PhenomXHMAmpInterpolMB = 1`` (see ``lalsim_src/LALSimIMRPhenomXHM_multiband.c``
and arXiv:2001.10897, García-Quirós & Husa 2020):

  * **amplitude** ``A(f)``: piecewise-**linear** interpolation of ``|h|`` between
    coarse points (GSL ``gsl_interp_linear``, ``ampinterpolorder = 1``).
  * **phase** ``φ(f)``: piecewise-**linear** interpolation of the **continuous
    (unwrapped)** phase between coarse points.  LAL builds each fine point via
    ``expφ = e^{iφ_j}·Q^{k}`` with ``Q = e^{i·Δf·Ω_j}`` and per-interval slope
    ``Ω_j = (φ_{j+1}−φ_j)/(f_{j+1}−f_j)`` (eq. 2.32).  That incremental product is
    exactly ``e^{i(φ_j + Ω_j·(f−f_j))}`` — i.e. linear interpolation of the
    continuous phase — so we evaluate it directly (more accurate, vectorised, and
    verified to match LAL's built-in multiband to overlap ~1e-8).

The reconstructed waveform is ``h = A · exp(i·φ)`` (the ``(−1)^ℓ`` prefactor is
+1 for the ℓ=2 dominant mode and is left to the caller for other modes).

Both interpolations use the existing, verified ``torch_linear_interp``
(``sage/core/interpolation.py``) — GSL-``gsl_interp_linear``-equivalent.

Two hard requirements:

  * **Continuous phase.** Between adjacent coarse multiband points the BNS phase
    winds through many cycles, so linear interpolation of the raw complex samples
    (or of the wrapped phase) is invalid.  Recover the continuous phase from a
    densely-sampled (Nyquist) full grid via :func:`continuous_coarse_phase`.
  * **float64.** The BNS continuous phase reaches ~1e4–1e5 rad; float32 (~7 sig.
    figs) cannot represent it.  Keep the coarse phase / fine grid in float64.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sage.core.interpolation import torch_linear_interp


# ── Phase unwrapping (torch-native, matches numpy.unwrap defaults) ────────────


def unwrap(phase: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Unwrap a phase tensor along ``dim`` by removing jumps > π, matching the
    default behaviour of ``numpy.unwrap`` (discontinuity = π, period = 2π).
    """
    dd = torch.diff(phase, dim=dim)
    two_pi = 2.0 * math.pi
    # Wrap each step into (−π, π].
    ddmod = torch.remainder(dd + math.pi, two_pi) - math.pi
    # numpy convention: a step of exactly −π that follows a positive raw diff is
    # mapped to +π so the correction follows the trend.
    ddmod = torch.where((ddmod == -math.pi) & (dd > 0),
                        torch.full_like(ddmod, math.pi), ddmod)
    correction = ddmod - dd
    # Only correct jumps larger than the discontinuity threshold (π).
    correction = torch.where(dd.abs() < math.pi,
                             torch.zeros_like(correction), correction)
    correction = correction.cumsum(dim=dim)
    pad_shape = list(phase.shape)
    pad_shape[dim] = 1
    zero = torch.zeros(pad_shape, dtype=correction.dtype, device=correction.device)
    correction = torch.cat([zero, correction], dim=dim)
    return phase + correction


def continuous_coarse_phase(
    h_full: torch.Tensor,
    coarse_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Continuous (unwrapped) phase of a full-grid FD waveform at the coarse points.

    Parameters
    ----------
    h_full : complex tensor, shape ``(..., F_full)``
        Waveform on the full uniform frequency grid (Nyquist-samples the phase,
        so unwrapping is unambiguous).  Use complex128 for BNS.
    coarse_indices : long tensor, shape ``(N_coarse,)``
        Indices of the coarse grid points into the full grid.

    Returns
    -------
    torch.Tensor
        Continuous phase (float64) at the coarse points, shape ``(..., N_coarse)``.
    """
    phi_full = unwrap(torch.angle(h_full), dim=-1)
    return phi_full[..., coarse_indices]


# ── Order-1 multiband reconstruction ──────────────────────────────────────────


def multiband_reconstruct(
    coarse_freqs: torch.Tensor,
    coarse_amp:   torch.Tensor,
    coarse_phase: torch.Tensor,
    fine_freqs:   torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct ``h(f) = A(f)·exp(i·φ(f))`` on ``fine_freqs`` from coarse samples,
    replicating LAL's order-1 multiband reconstruction (linear amplitude, linear
    continuous phase) via :func:`sage.core.interpolation.torch_linear_interp`.

    Parameters
    ----------
    coarse_freqs : (N_coarse,) float64, strictly increasing.
    coarse_amp   : (N_coarse,) float64 — amplitude ``|h|`` at coarse points.
    coarse_phase : (N_coarse,) float64 — **continuous (unwrapped)** phase.
    fine_freqs   : (N_fine,)   float64 — target grid within
        ``[coarse_freqs[0], coarse_freqs[-1]]``.

    Returns
    -------
    torch.Tensor
        Complex128 reconstruction, shape ``(N_fine,)``.
    """
    a   = torch_linear_interp(fine_freqs, coarse_freqs, coarse_amp)    # linear |h|
    phi = torch_linear_interp(fine_freqs, coarse_freqs, coarse_phase)  # linear cont. phase
    return torch.polar(a, phi)                                         # A · exp(i·φ)


class MultibandReconstructor(nn.Module):
    """
    Order-1 multiband reconstruction onto a fixed fine grid (thin wrapper around
    :func:`multiband_reconstruct` that stores the two frequency grids as buffers).
    """

    def __init__(self, coarse_freqs: torch.Tensor, fine_freqs: torch.Tensor):
        super().__init__()
        self.register_buffer("coarse_freqs", coarse_freqs.to(torch.float64))
        self.register_buffer("fine_freqs",   fine_freqs.to(torch.float64))

    def forward(self, coarse_amp: torch.Tensor, coarse_phase: torch.Tensor) -> torch.Tensor:
        return multiband_reconstruct(
            self.coarse_freqs, coarse_amp, coarse_phase, self.fine_freqs
        )
