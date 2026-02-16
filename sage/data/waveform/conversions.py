#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : conversions.py
Description     : Short description of the file

Created on 2026-02-16 16:05:31

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
import torch


def mass1_mass2_to_mchirp_q(m1, m2):
    q = m1 / m2
    mchirp = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
    return mchirp, q


def chirp_distance_to_distance(chirp_distance, mchirp):
    return chirp_distance * (mchirp / 1.2) ** (5 / 6)
