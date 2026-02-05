#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : math.py
Description     : Short description of the file

Created on 2026-01-21 07:59:05

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
import numpy as np


class Normalise:
    """
    Normalise the variable using known bounds

        For example, norm_tc = (tc - min_val)/(max_val - min_val)
        The values of max_val and min_val are provided
        to the class. obj.norm can be called during
        data generation to get normalised values of tc, if needed.

    """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def norm(self, val):
        # Return normalised value in [0, 1]
        return (val - self.min_val) / (self.max_val - self.min_val)

    def unnorm(self, val):
        # Return unnormalised value in original scale
        return (val * (self.max_val - self.min_val)) + self.min_val


class Standardise:
    """
    Standardise a variable to zero mean and unit variance.
    """

    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def norm(self, val):
        # Return zero mean unit variance output
        return (val - self.mean) / (self.std + self.eps)

    def unnorm(self, val):
        # Return original scale output
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
