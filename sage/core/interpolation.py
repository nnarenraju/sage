#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : interpolation.py
Description     : Short description of the file

Created on 2026-01-23 02:36:43

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
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
    1D linear interpolation, compatible with ``jnp.interp`` / ``np.interp``.

    Interpolation is over the LAST axis of ``fp``, so ``fp`` may carry leading
    batch dimensions: ``fp`` of shape ``(..., N)`` is interpolated at the query
    points ``x`` (shape ``(M,)``) to give ``(..., M)``.  The 1-D case
    (``fp`` shape ``(N,)``) is unchanged.

    Parameters
    ----------
    x : torch.Tensor, shape ``(M,)`` (or ``(...,)`` for scalar-batch ``fp``)
        Query points.
    xp : torch.Tensor, shape ``(N,)``
        Monotonically increasing node x-coordinates.
    fp : torch.Tensor, shape ``(..., N)``
        Function values at ``xp`` (last axis), with optional leading batch dims.

    Returns
    -------
    torch.Tensor, shape ``(..., M)``
        Linearly interpolated values at ``x``.
    """
    # indices where elements should be inserted
    idx = torch.searchsorted(xp, x, right=True)

    idx = torch.clamp(idx, 1, xp.numel() - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[..., idx - 1]      # gather over the last axis (batched-safe)
    y1 = fp[..., idx]

    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0)


@torch.compile
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
    Second derivatives of the natural cubic spline through ``(xp, fp)``.

    The interpolant of GSL's ``gsl_interp_cspline`` ("cubic spline with
    natural boundary conditions"), which LAL uses for the PhenomD QNM tables.
    The second derivatives vanish at both ends, and the interior ones solve
    the tridiagonal system of Press et al., Numerical Recipes 3rd ed., eq. 3.3.7,

        h[i-1] M[i-1] + 2 (h[i-1] + h[i]) M[i] + h[i] M[i+1]
            = 6 ((fp[i+1] - fp[i]) / h[i] - (fp[i] - fp[i-1]) / h[i-1]),

    with ``h[i] = xp[i+1] - xp[i]``, by the tridiagonal algorithm of §2.4.
    Leading dimensions are independent splines.

    Parameters
    ----------
    xp : torch.Tensor, shape ``(..., N)``
        Strictly increasing nodes.
    fp : torch.Tensor, shape ``(..., N)``
        Values at the nodes.

    Returns
    -------
    torch.Tensor, shape ``(..., N)``
        Second derivatives at the nodes, zero at both ends.
    """
    h = xp[..., 1:] - xp[..., :-1]
    diag = 2 * (h[..., :-1] + h[..., 1:])
    off = h[..., 1:-1]
    rhs = 6 * (
        (fp[..., 2:] - fp[..., 1:-1]) / h[..., 1:]
        - (fp[..., 1:-1] - fp[..., :-2]) / h[..., :-1]
    )

    # Forward elimination of the sub-diagonal, then back substitution.
    n = diag.shape[-1]
    for i in range(1, n):
        weight = off[..., i - 1] / diag[..., i - 1]
        diag[..., i] = diag[..., i] - weight * off[..., i - 1]
        rhs[..., i] = rhs[..., i] - weight * rhs[..., i - 1]

    M = torch.zeros_like(fp)
    interior = torch.zeros_like(rhs)
    interior[..., -1] = rhs[..., -1] / diag[..., -1]
    for i in range(n - 2, -1, -1):
        numerator = rhs[..., i] - off[..., i] * interior[..., i + 1]
        interior[..., i] = numerator / diag[..., i]
    M[..., 1:-1] = interior
    return M


def torch_natural_cubic_interp(x, xp, fp, M, derivative=False):
    """
    Natural cubic spline through ``(xp, fp)``, or its derivative, at ``x``.

    Press et al., Numerical Recipes 3rd ed., eq. 3.3.3 for the value and
    eq. 3.3.5 for the derivative; with ``M`` from ``torch_natural_cubic_coeffs``
    this is what ``gsl_spline_eval`` returns for ``gsl_interp_cspline``.

    Parameters
    ----------
    x : torch.Tensor
        Query points within ``[xp[0], xp[-1]]``: any shape for 1-D nodes,
        shape ``(..., K)`` for batched nodes.
    xp : torch.Tensor, shape ``(N,)`` or ``(..., N)``
        Strictly increasing nodes.
    fp : torch.Tensor, shape like ``xp``
        Values at the nodes.
    M : torch.Tensor, shape like ``xp``
        Second derivatives at the nodes, from ``torch_natural_cubic_coeffs``.
    derivative : bool
        Return the first derivative instead of the value.

    Returns
    -------
    torch.Tensor
        Spline values or derivatives at ``x``.
    """
    idx = torch.searchsorted(xp, x.contiguous(), right=True) - 1
    idx = idx.clamp(0, xp.shape[-1] - 2)

    x_i = _take(xp, idx)
    x_ip1 = _take(xp, idx + 1)
    h = x_ip1 - x_i

    y_i = _take(fp, idx)
    y_ip1 = _take(fp, idx + 1)

    M_i = _take(M, idx)
    M_ip1 = _take(M, idx + 1)

    a = (x_ip1 - x) / h
    b = (x - x_i) / h

    if not derivative:
        return (
            a * y_i + b * y_ip1 + ((a**3 - a) * M_i + (b**3 - b) * M_ip1) * (h**2) / 6
        )

    # derivative
    return (y_ip1 - y_i) / h + h * ((3 * b * b - 1) * M_ip1 - (3 * a * a - 1) * M_i) / 6


def _take(values, idx):
    """
    Entries of ``values`` at ``idx`` along the last axis.

    Parameters
    ----------
    values : torch.Tensor, shape ``(N,)`` or ``(..., N)``
    idx : torch.Tensor of int
        Any shape for 1-D ``values``, ``(..., K)`` otherwise.

    Returns
    -------
    torch.Tensor, shape of ``idx``
    """
    return values[idx] if values.dim() == 1 else values.gather(-1, idx)
