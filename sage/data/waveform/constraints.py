#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : constraints.py
Description     : Short description of the file

Created on 2026-02-16 14:28:00

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
import sys
import torch
import inspect


def mass_order(params):
    """Ensure mass1 >= mass2 by swapping where needed."""
    m1 = params["mass1"]
    m2 = params["mass2"]

    swap_mask = m2 > m1

    if swap_mask.any():
        m1_new = m1.clone()
        m2_new = m2.clone()
        m1_new[swap_mask] = m2[swap_mask]
        m2_new[swap_mask] = m1[swap_mask]

        params["mass1"] = m1_new
        params["mass2"] = m2_new

    return params


## For automatically adding all named constraints

_current_module = sys.modules[__name__]

_NAMED_CONSTRAINTS = [
    name
    for name, obj in inspect.getmembers(_current_module, inspect.isfunction)
    if obj.__module__ == __name__ and not name.startswith("_")
]
