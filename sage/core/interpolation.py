#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interpolation in torch: linear, Catmull-Rom on a uniform grid, and
the natural cubic spline of GSL's ``gsl_interp_cspline``.

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

import torch


def torch_linear_interp(x, xp, fp):
    """
    Linear interpolation, as ``np.interp``, along the last axis.

    ``fp`` of shape ``(..., N)`` at ``x`` of shape ``(M,)`` gives
    ``(..., M)``.

    Parameters
    ----------
    x : torch.Tensor, shape ``(M,)``
        Query points.
    xp : torch.Tensor, shape ``(N,)``
        Increasing nodes.
    fp : torch.Tensor, shape ``(..., N)``
        Values at ``xp`` along the last axis.

    Returns
    -------
    torch.Tensor, shape ``(..., M)``
        Interpolated values.
    """
    # Right neighbour of each query point
    idx = torch.searchsorted(xp, x, right=True)

    idx = torch.clamp(idx, 1, xp.numel() - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[..., idx - 1]      # last axis, any leading dimensions
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
    Catmull-Rom cubic interpolation on a uniform grid.

    Parameters
    ----------
    xs : torch.Tensor
        Query points, any shape.
    y : torch.Tensor, shape ``(N,)``
        Values on the grid.
    x0 : float
        First grid point.
    dx : float
        Grid spacing.

    Returns
    -------
    torch.Tensor, shape of ``xs``
        Interpolated values.
    """

    # Continuous index
    t = (xs - x0) / dx

    i = torch.floor(t).long()

    i = torch.clamp(i, 1, y.shape[0] - 3)

    # Local coordinate
    u = t - i

    y0 = y[i - 1]
    y1 = y[i]
    y2 = y[i + 1]
    y3 = y[i + 2]

    # Catmull-Rom coefficients
    a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
    b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    c = -0.5 * y0 + 0.5 * y2
    d = y1

    return ((a * u + b) * u + c) * u + d


def torch_natural_cubic_coeffs(xp, fp):
    """
    Second derivatives of the natural cubic spline through ``(xp, fp)``.

    As GSL's ``gsl_interp_cspline``: zero at both ends, interior from
    the tridiagonal system for continuous first derivatives. Leading
    dimensions are separate splines.

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
    Natural cubic spline through ``(xp, fp)``, or its derivative.

    With ``M`` from ``torch_natural_cubic_coeffs`` this is
    ``gsl_spline_eval`` for ``gsl_interp_cspline``.

    Parameters
    ----------
    x : torch.Tensor
        Query points within ``[xp[0], xp[-1]]``; any shape for 1-D
        nodes, ``(..., K)`` for batched nodes.
    xp : torch.Tensor, shape ``(N,)`` or ``(..., N)``
        Strictly increasing nodes.
    fp : torch.Tensor, shape like ``xp``
        Values at the nodes.
    M : torch.Tensor, shape like ``xp``
        Second derivatives at the nodes, from
        ``torch_natural_cubic_coeffs``.
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

    return (y_ip1 - y_i) / h + h * ((3 * b * b - 1) * M_ip1 - (3 * a * a - 1) * M_i) / 6


def _take(values, idx):
    """
    Entries of ``values`` at ``idx`` along the last axis.

    Parameters
    ----------
    values : torch.Tensor, shape ``(N,)`` or ``(..., N)``
        Values along the last axis.
    idx : torch.Tensor of int
        Any shape for 1-D ``values``, ``(..., K)`` otherwise.

    Returns
    -------
    torch.Tensor, shape of ``idx``
    """
    return values[idx] if values.dim() == 1 else values.gather(-1, idx)
