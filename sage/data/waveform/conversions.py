#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : conversions.py
Description     : Short description of the file

Created on 2026-02-16 16:05:31

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
import torch

from sage.core.conversions import mchirp_from_mass1_mass2


def mass1_mass2_to_mchirp_q(m1, m2):
    """
    Convert component masses to chirp mass and mass ratio.

    Parameters
    ----------
    m1 : float or torch.Tensor
        Primary mass (must be >= m2).
    m2 : float or torch.Tensor
        Secondary mass.

    Returns
    -------
    mchirp : same type as input
        Chirp mass ``eta^(3/5) (m1 + m2)``, as ``pycbc.conversions.mchirp_from_mass1_mass2``.
    q : same type as input
        Mass ratio ``m1 / m2`` (>= 1 when m1 >= m2).
    """
    q = m1 / m2
    mchirp = mchirp_from_mass1_mass2(m1, m2)
    return mchirp, q
