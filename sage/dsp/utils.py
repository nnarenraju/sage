#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Edge trimming of filtered or resampled time series.

Filename        : utils.py
Description     : Short description of the file

Created on 2025-12-12 15:49:41

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

from sage.core.logger import get_logger

logger = get_logger(__name__)


def trim_edges(data, fs, trim=0.2):
    """
    Remove ``trim`` seconds from each end of a time series.

    Parameters
    ----------
    data : array-like, shape ``(N,)``
        Time series to trim.
    fs : float
        Sampling rate in Hz.
    trim : float
        Seconds to remove from each end.

    Returns
    -------
    array-like
        Trimmed series of length ``N - 2 * round(trim * fs)``.

    Raises
    ------
    ValueError
        If the trim is zero samples or at least half the data length.
    """

    n = int(round(trim * fs))

    if n == 0 or 2 * n >= len(data):
        logger.error("Trim too large for data length.")
        raise ValueError("Trim too large for data length.")

    return data[n:-n]