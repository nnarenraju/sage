#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : conversions.py
Description     : Short description of the file

Created on 2026-01-20 16:26:05

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


def seconds_to_samples(nseconds, sample_rate, approx_mode=int, rounding=True):
    if rounding:
        # No need to change the base for rounding
        return approx_mode(np.around(nseconds * sample_rate))
    else:
        return approx_mode(nseconds * sample_rate)


def samples_to_seconds(nsamples, sample_rate):
    return nsamples / sample_rate
