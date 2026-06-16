#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lazy access to pycbc constants for the noise package.

pycbc is an *optional* dependency (declared under the ``lal`` extra in
``pyproject.toml``). Several noise modules only need pycbc's dynamic-range
constant, and only when actually reading real noise — not at import time. Routing
that constant through this helper keeps ``import sage.data.noise`` working in a
pycbc-free environment (e.g. the lightweight CI), deferring the heavy import to
the first noise read.
"""

_DYN_RANGE_FAC = None


def dyn_range_fac():
    """Return pycbc's ``DYN_RANGE_FAC``, importing pycbc lazily on first use."""
    global _DYN_RANGE_FAC
    if _DYN_RANGE_FAC is None:
        from pycbc import DYN_RANGE_FAC
        _DYN_RANGE_FAC = DYN_RANGE_FAC
    return _DYN_RANGE_FAC
