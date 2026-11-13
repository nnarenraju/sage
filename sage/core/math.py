#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Numerical helpers: gradient-safe square roots, a glibc-exact cube
root, min-max and z-score scalers, and axis rotation matrices.

Filename        : math.py
Description     : Short description of the file

Created on 2026-01-21 07:59:05

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

import numpy as np
import torch


def safe_sqrt(x):
    """
    ``sqrt(x)`` with a zero, not infinite, derivative at ``x = 0``.

    The root is taken on an operand masked away from zero, so values
    are bit-identical to :func:`torch.sqrt` for ``x >= 0``. Used where
    an attainable limit sits on the branch point, e.g. ``cos beta = 1``.

    Parameters
    ----------
    x : torch.Tensor
        Non-negative operand.

    Returns
    -------
    torch.Tensor
        ``sqrt(x)``, with zero gradient contribution where ``x == 0``.
    """
    nonzero = x > 0
    return torch.where(
        nonzero,
        torch.sqrt(torch.where(nonzero, x, torch.ones_like(x))),
        torch.zeros_like(x),
    )


class _StrongZeroSqrt(torch.autograd.Function):
    """``sqrt`` whose backward is 0 where the incoming gradient is 0."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        Square root of ``x``, saved for the backward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context.
        x : torch.Tensor
            Non-negative operand.

        Returns
        -------
        torch.Tensor
            ``sqrt(x)``.
        """
        result = torch.sqrt(x)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        """
        Gradient ``grad / (2 sqrt(x))``, and 0 where ``grad`` is 0.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context holding ``sqrt(x)``.
        grad : torch.Tensor
            Gradient with respect to the output.

        Returns
        -------
        torch.Tensor
            Gradient with respect to ``x``.
        """
        (result,) = ctx.saved_tensors
        return torch.where(grad == 0, torch.zeros_like(grad), grad / (2.0 * result))


def strong_zero_sqrt(x: torch.Tensor) -> torch.Tensor:
    """
    ``sqrt(x)`` with a zero gradient where the incoming gradient is 0.

    Forward values are those of :func:`torch.sqrt`.  At ``x = 0`` an
    unused result passes 0 rather than 0 * inf = NaN; a used result
    still passes the infinite derivative.

    Parameters
    ----------
    x : torch.Tensor
        Non-negative operand.

    Returns
    -------
    torch.Tensor
        ``sqrt(x)``.
    """
    return _StrongZeroSqrt.apply(x)


def detach_where(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    ``x`` with the entries under ``mask`` detached from the graph.

    Forward values are unchanged.  Applied to the inputs of a branch
    that is evaluated everywhere but used only off ``mask``, it keeps
    that branch from passing gradient (or 0 * inf = NaN) under ``mask``.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean mask, broadcastable against ``x``.
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        ``x`` broadcast against ``mask``, or ``x`` itself when it does
        not require grad.
    """
    if not x.requires_grad:
        return x
    return torch.where(mask, x.detach(), x)


# glibc 2.34 sysdeps/ieee754/dbl-64/s_cbrt.c
_CBRT2 = 1.2599210498948731648
_SQR_CBRT2 = 1.5874010519681994748
_CBRT_FACTOR = (1.0 / _SQR_CBRT2, 1.0 / _CBRT2, 1.0, _CBRT2, _SQR_CBRT2)
_CBRT_POLY = (
    -1.83469277483613086,
    2.44693122563534430,
    -2.11499494167371287,
    1.50819193781584896,
    0.354895765043919860,
)


class _Cbrt(torch.autograd.Function):
    """glibc ``cbrt`` with the backward of ``x ** (1/3)``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        glibc ``cbrt``; zero and non-finite entries return ``x + x``.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context; ``x`` is saved for the backward pass.
        x : torch.Tensor
            Float64 operand.

        Returns
        -------
        torch.Tensor
            ``cbrt(x)``.
        """
        ctx.save_for_backward(x)
        xm, xe = torch.frexp(torch.abs(x))
        u = 0.784932344976639262 - 0.145263899385486377 * xm
        for coefficient in _CBRT_POLY:
            u = coefficient + u * xm
        t2 = u * u * u
        factor = torch.tensor(_CBRT_FACTOR, dtype=x.dtype, device=x.device)
        factor = factor[2 + torch.fmod(xe, 3).long()]
        ym = u * (t2 + 2.0 * xm) / (2.0 * t2 + xm) * factor
        exponent = torch.div(xe, 3, rounding_mode="trunc")
        y = torch.ldexp(torch.where(x > 0.0, ym, -ym), exponent)
        return torch.where((x == 0.0) | ~torch.isfinite(x), x + x, y)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        """
        Gradient of ``x ** (1/3)``, ``grad * x ** (-2/3) / 3``.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context holding ``x``.
        grad : torch.Tensor
            Gradient with respect to the output.

        Returns
        -------
        torch.Tensor
            Gradient with respect to ``x``.
        """
        (x,) = ctx.saved_tensors
        return grad * ((1.0 / 3.0) * x.pow(1.0 / 3.0 - 1.0))


def cbrt(x: torch.Tensor) -> torch.Tensor:
    """
    Cube root, bit-identical to the glibc 2.34 ``cbrt`` that LAL calls.

    Uses only float64 frexp, ldexp and arithmetic.

    Parameters
    ----------
    x : torch.Tensor
        Float64 operand.

    Returns
    -------
    torch.Tensor
        ``cbrt(x)``.
    """
    return _Cbrt.apply(x)


class Normalise:
    """
    Min-max normalisation to the unit interval ``[0, 1]``.

    Maps ``min_val`` to 0 and ``max_val`` to 1; :meth:`unnorm` inverts
    the map.

    Parameters
    ----------
    min_val : float
        Lower bound of the original parameter range.
    max_val : float
        Upper bound of the original parameter range.
    """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def norm(self, val):
        """
        Normalise ``val`` to ``[0, 1]``.

        Parameters
        ----------
        val : float or array-like
            Value on the original scale.

        Returns
        -------
        float or array-like
            ``(val - min_val) / (max_val - min_val)``.
        """
        return (val - self.min_val) / (self.max_val - self.min_val)

    def unnorm(self, val):
        """
        Map a normalised value back to the original scale.

        Parameters
        ----------
        val : float or array-like
            Normalised value.

        Returns
        -------
        float or array-like
            ``val * (max_val - min_val) + min_val``.
        """
        return (val * (self.max_val - self.min_val)) + self.min_val


class Standardise:
    """
    Z-score standardisation to zero mean and unit variance.

    ``eps`` is added to ``std`` to avoid division by zero for a
    constant parameter; :meth:`unnorm` inverts the map.

    Parameters
    ----------
    mean : float
        Population mean of the parameter.
    std : float
        Population standard deviation of the parameter.
    eps : float
        Added to ``std`` in both directions (default 1e-8).
    """

    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def norm(self, val):
        """
        Standardise ``val`` to zero mean and unit variance.

        Parameters
        ----------
        val : float or array-like
            Value on the original scale.

        Returns
        -------
        float or array-like
            ``(val - mean) / (std + eps)``.
        """
        return (val - self.mean) / (self.std + self.eps)

    def unnorm(self, val):
        """
        Map a standardised value back to the original scale.

        Parameters
        ----------
        val : float or array-like
            Standardised value.

        Returns
        -------
        float or array-like
            ``val * (std + eps) + mean``.
        """
        return val * (self.std + self.eps) + self.mean


# Adapted from astropy.coordinates.matrix_utilities.rotation_matrix
def rotation_matrix(angle_in_rad, axis=2):
    """
    Matrix of a rotation by a scalar angle about the x, y or z axis.

    Parameters
    ----------
    angle_in_rad : float
        Rotation angle in radians.
    axis : int
        Rotation axis: 0 (x), 1 (y) or 2 (z).

    Returns
    -------
    numpy.ndarray
        The 3 x 3 rotation matrix.

    Raises
    ------
    ValueError
        If ``axis`` is not 0, 1 or 2.
    """

    if axis not in (0, 1, 2):
        raise ValueError("Axis must be 0 (x), 1 (y), or 2 (z)")

    s = np.sin(angle_in_rad)
    c = np.cos(angle_in_rad)

    R = np.zeros((3, 3), dtype=float)

    a1 = (axis + 1) % 3
    a2 = (axis + 2) % 3
    R[..., axis, axis] = 1.0
    R[..., a1, a1] = c
    R[..., a1, a2] = s
    R[..., a2, a1] = -s
    R[..., a2, a2] = c

    return R
