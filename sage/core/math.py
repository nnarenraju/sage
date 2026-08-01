#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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

# Packages
import numpy as np


class Normalise:
    """
    Min-max normalisation to the unit interval ``[0, 1]``.

    Maps ``val`` linearly so that ``min_val`` → 0 and ``max_val`` → 1.
    Provides a symmetric :meth:`unnorm` inverse.  Used by
    :class:`~sage.data.waveform.sampler.DistributionSampler` to normalise
    waveform parameters to a common scale before regression targets are
    passed to the network.

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
        """Normalise ``val`` to ``[0, 1]``."""
        return (val - self.min_val) / (self.max_val - self.min_val)

    def unnorm(self, val):
        """Invert normalisation, recovering the original scale."""
        return (val * (self.max_val - self.min_val)) + self.min_val


class Standardise:
    """
    Z-score standardisation to zero mean and unit variance.

    Maps ``val`` so that the distribution has mean 0 and std 1.  The small
    ``eps`` guard prevents division by zero for constant-valued parameters.

    Parameters
    ----------
    mean : float
        Population mean of the parameter.
    std : float
        Population standard deviation of the parameter.
    eps : float
        Numerical stability guard (default 1e-8).
    """

    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def norm(self, val):
        """Standardise ``val`` to zero mean, unit variance."""
        return (val - self.mean) / (self.std + self.eps)

    def unnorm(self, val):
        """Invert standardisation, recovering the original scale."""
        return val * (self.std + self.eps) + self.mean


# Refer: https://docs.astropy.org/en/stable/_modules/astropy/coordinates/matrix_utilities.html#rotation_matrix
def rotation_matrix(angle_in_rad, axis=2):
    """
    Generate matrices for rotation by some angle around some axis.
    This version ONLY supports x,y,z axes; general axis version removed

    Parameters
    ----------
    angle : angle-like
        The amount of rotation the matrices should represent.  Can be an array.
    axis : int
        Only x,y,z supported. {x,y,z} -> {0,1,2}

    Returns
    -------
    rmat : torch.tensor
        A unitary rotation matrix.
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
