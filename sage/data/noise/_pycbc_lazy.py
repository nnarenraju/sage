#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lazy access to pycbc's ``DYN_RANGE_FAC`` for the noise package.

pycbc is imported on the first call, not when ``sage.data.noise`` is
imported.
"""

_DYN_RANGE_FAC = None


def dyn_range_fac():
    """
    pycbc's ``DYN_RANGE_FAC``, importing pycbc on the first call.

    Returns
    -------
    float
        The dynamic-range factor.
    """
    global _DYN_RANGE_FAC
    if _DYN_RANGE_FAC is None:
        # Deferred: importing the noise package does not load pycbc
        from pycbc import DYN_RANGE_FAC
        _DYN_RANGE_FAC = DYN_RANGE_FAC
    return _DYN_RANGE_FAC
