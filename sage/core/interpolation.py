#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : interpolation.py
Description     : Short description of the file

Created on 2026-01-23 02:36:43

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
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
import torch


def torch_linear_interp(x, xp, fp):
    """
    1D linear interpolation like jnp.interp.
    x:  Tensor (...,)
    xp: Tensor (N,) increasing
    fp: Tensor (N,)
    """
    # indices where elements should be inserted
    idx = torch.searchsorted(xp, x, right=True)

    idx = torch.clamp(idx, 1, xp.numel() - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]

    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0)


def torch_scipylike_cubic_interp(x, xp, fp):
    """
    1D Cubic interpolation like scipy.interpolate.CubicSpline
    x:  Tensor (...,)
    xp: Tensor (N,) increasing
    fp: Tensor (N,)
    """
    idx = torch.searchsorted(xp, x, right=True)
    idx = idx.clamp(1, xp.numel() - 2)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    x2 = xp[idx + 1]

    y0 = fp[idx - 1]
    y1 = fp[idx]
    y2 = fp[idx + 1]

    # Finite-difference slopes
    m1 = (y1 - y0) / (x1 - x0)
    m2 = (y2 - y1) / (x2 - x1)

    t = (x - x1) / (x2 - x1)

    # Hermite basis
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)

    return h00 * y1 + h10 * (x2 - x1) * m1 + h01 * y2 + h11 * (x2 - x1) * m2


@torch.jit.script
def torch_catmull_rom_cubic_interp(
    xs: torch.Tensor,
    y: torch.Tensor,
    x0: float,
    dx: float,
):
    """
    Fast cubic interpolation on a *uniform grid* (Catmull-Rom).

    Args:
        xs: (...,) query points
        y:  (N,) sampled values on uniform grid
        x0: grid start
        dx: grid spacing

    Returns:
        (...,) interpolated values
    """

    # Continuous index
    t = (xs - x0) / dx

    # Left index
    i = torch.floor(t).long()

    # Clamp to valid range
    i = torch.clamp(i, 1, y.shape[0] - 3)

    # Local coordinate
    u = t - i

    # Fetch points
    y0 = y[i - 1]
    y1 = y[i]
    y2 = y[i + 1]
    y3 = y[i + 2]

    # Catmull–Rom coefficients
    a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
    b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    c = -0.5 * y0 + 0.5 * y2
    d = y1

    return ((a * u + b) * u + c) * u + d


def torch_natural_cubic_coeffs(xp, fp):
    """
    Compute second derivatives for a natural cubic spline.
    Equivalent to gsl_interp_cspline.

    xp: (N,) strictly increasing
    fp: (N,)
    returns: M (N,) second derivatives
    """
    N = xp.numel()
    device = xp.device
    dtype = xp.dtype

    h = xp[1:] - xp[:-1]  # (N-1,)

    # RHS
    rhs = torch.zeros(N, device=device, dtype=dtype)
    rhs[1:-1] = 6 * ((fp[2:] - fp[1:-1]) / h[1:] - (fp[1:-1] - fp[:-2]) / h[:-1])

    # Tridiagonal matrix
    lower = torch.zeros(N - 1, device=device, dtype=dtype)
    diag = torch.ones(N, device=device, dtype=dtype)
    upper = torch.zeros(N - 1, device=device, dtype=dtype)

    diag[0] = diag[-1] = 1.0  # natural BC
    diag[1:-1] = 2 * (h[:-1] + h[1:])
    lower[1:] = h[:-1]
    upper[:-1] = h[1:]

    # Solve tridiagonal system (Thomas algorithm)
    # Forward sweep
    for i in range(1, N):
        w = lower[i - 1] / diag[i - 1]
        diag[i] -= w * upper[i - 1]
        rhs[i] -= w * rhs[i - 1]

    # Back substitution
    M = torch.zeros(N, device=device, dtype=dtype)
    M[-1] = rhs[-1] / diag[-1]
    for i in range(N - 2, -1, -1):
        M[i] = (rhs[i] - upper[i] * M[i + 1]) / diag[i]

    return M


def torch_natural_cubic_interp(x, xp, fp, M, derivative=False):
    """
    Compute a natural cubic spline interpolation of fp at points x using nodes xp.
    Matches gsl_spline_eval from LAL as much as possible.

    Args:
        x (Tensor): Points where the interpolated values are desired (...,).
        xp (Tensor): Monotonically increasing node points (N,).
        fp (Tensor): Function values at nodes xp (N,).
        M (float or Tensor): Total mass scaling factor,
            used to convert to physical units if needed.

    Returns:
        Tensor: Interpolated values at x using a natural cubic spline.

    # NOTE:
    # LAL computes certain derivatives using a *natural cubic spline*
    # (gsl_interp_cspline), which enforces global C2 smoothness and
    # zero second derivatives at the endpoints.
    #
    # Say we look at the example of enforcing time of coalescence at t=0.
    # The phase is differentiated to compute a time shift that enforces coalescence
    # at t = 0. Using a local or non-natural cubic interpolant changes dPhase/df at
    # f_final. This produces an incorrect time shift, resulting in a constant
    # phase error that propagates across the entire waveform
    # (inspiral, merger, ringdown).
    #
    # To remain consistent with the C/LAL implementation, the same natural cubic
    # spline must be used here.

    Example usage:
    M = torch_natural_cubic_spline_coeffs(freqs_fixed, phase_fixed)
    phi_interp = lambda f: torch_natural_cubic_interp(f, freqs_fixed, phase_fixed, M)
    t_corr = torch_grad(phi_interp, (f_final,)) / (2 * torch.pi)

    """
    idx = torch.searchsorted(xp, x, right=True) - 1
    idx = idx.clamp(0, xp.numel() - 2)

    x_i = xp[idx]
    x_ip1 = xp[idx + 1]
    h = x_ip1 - x_i

    y_i = fp[idx]
    y_ip1 = fp[idx + 1]

    M_i = M[idx]
    M_ip1 = M[idx + 1]

    a = (x_ip1 - x) / h
    b = (x - x_i) / h

    if not derivative:
        return (
            a * y_i + b * y_ip1 + ((a**3 - a) * M_i + (b**3 - b) * M_ip1) * (h**2) / 6
        )

    # derivative
    return (y_ip1 - y_i) / h + h * ((3 * b * b - 1) * M_ip1 - (3 * a * a - 1) * M_i) / 6
