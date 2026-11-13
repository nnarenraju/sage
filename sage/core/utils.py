#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Argument normalisation: values as tuples and as one-dimensional
arrays.

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

import numpy as np


def to_sequence(x):
    """
    ``x`` as a tuple; a scalar, string or bytes value becomes a 1-tuple.

    Parameters
    ----------
    x : scalar, str, bytes, iterable or None
        Value to wrap.

    Returns
    -------
    tuple or None
        ``None`` when ``x`` is ``None``.
    """
    if x is None:
        return x
    if isinstance(x, (str, bytes)):
        return (x,)
    if isinstance(x, (int, float)):
        return (x,)
    if isinstance(x, (bool)):
        return (x,)
    return tuple(x)


def ensure_1d(x):
    """
    ``x`` as a NumPy array, which must be one-dimensional.

    Parameters
    ----------
    x : array-like
        Input to validate.

    Returns
    -------
    numpy.ndarray, shape ``(N,)``
        ``x`` as an array.

    Raises
    ------
    ValueError
        If the array is not one-dimensional.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input must be 1D")
    return x
