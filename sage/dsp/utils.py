#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : utils.py
Description     : Short description of the file

Created on 2025-12-12 15:49:41

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# LOCAL
from sage.core.logger import get_logger

logger = get_logger(__name__)


def trim_edges(data, fs, trim=0.2):
    """
    Trim corrupted edge samples from a filtered or resampled time series.

    Digital filters and resampling operations introduce transient artefacts
    at the boundaries of a segment.  This function removes ``trim`` seconds
    from each end to discard those corrupted samples.

    Parameters
    ----------
    data : array-like, shape ``(N,)``
        1D time series to trim.
    fs : float
        Sampling rate in Hz.
    trim : float
        Number of seconds to remove from each end (default 0.2 s).

    Returns
    -------
    numpy.ndarray
        Trimmed time series of length ``N - 2 * round(trim * fs)``.

    Raises
    ------
    ValueError
        If the trim length equals or exceeds half the data length.
    """

    n = int(round(trim * fs))

    if n == 0 or 2 * n >= len(data):
        logger.error("Trim too large for data length.")
        raise ValueError("Trim too large for data length.")

    return data[n:-n]