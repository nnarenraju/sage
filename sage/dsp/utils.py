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
    """Trim data edges after filtering/resampling.

    Args:
        data (array): 1D time series
        fs (float): Sampling rate (Hz)
        trim (float, optional): Edge trim (seconds) for normal mode.
            - Defaults to 0.2

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    n = int(round(trim * fs))

    if n == 0 or 2 * n >= len(data):
        logger.error("Trim too large for data length.")
        raise ValueError("Trim too large for data length.")

    return data[n:-n]