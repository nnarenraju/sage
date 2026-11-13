#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Amplitude spectral density estimation, smoothing, line handling and
reading of the fiducial ASDs.

Filename        : __init__.py
Description     : Amplitude spectral density estimation, smoothing and I/O.

Created on 2026-03-02 22:53:53

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL


Note on naming
--------------
Sage works in the ASD domain throughout. The Welch estimator in
:mod:`sage.dsp.welch` returns a PSD and
:class:`~sage.data.asd.get_asds.EstimateASD` takes its square root once;
every stored spectrum (fiducial, recolour bank, per-segment bank) is an
ASD in strain/sqrt(Hz). Files on disk keep their ``*_psd*`` names so
that existing data releases and run exports remain readable.
"""

from .read_asds import fiducial_asds_from_record, read_fiducial_asds

__all__ = ["fiducial_asds_from_record", "read_fiducial_asds"]
