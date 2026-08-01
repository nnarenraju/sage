#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : utils.py
Description     : Short description of the file

Created on 2025-11-06 17:39:59

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
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


def to_sequence(x):
    """Wrap scalars into a tuple so everything becomes iterable."""
    if x is None:
        return x
    if isinstance(x, (str, bytes)):
        return (x,)  # single string as a tuple
    if isinstance(x, (int, float)):
        return (x,)
    if isinstance(x, (bool)):
        return (x,)
    return tuple(x)


def ensure_1d(x):
    """
    Convert ``x`` to a NumPy array and assert it is one-dimensional.

    Parameters
    ----------
    x : array-like
        Input to validate.

    Returns
    -------
    numpy.ndarray, shape ``(N,)``
        One-dimensional view of ``x``.

    Raises
    ------
    ValueError
        If the resulting array has more than one dimension.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input must be 1D")
    return x
