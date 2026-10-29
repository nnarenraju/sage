#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : constraints.py
Description     : Short description of the file

Created on 2026-02-16 14:28:00

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
import sys
import torch
import inspect


def mass_order(params, param_index, extra_params=None):
    """Samples with ``mass1 >= mass2`` (PyCBC's constraint), as a boolean mask."""
    return params[:, param_index["mass1"]] >= params[:, param_index["mass2"]]


def lambda_order(params, param_index, extra_params=None):
    """
    Samples with ``lambda1 <= lambda2``, as a boolean mask.

    Astrophysically the more massive component (mass1, after
    :func:`mass_order`) is more compact and has the SMALLER tidal
    deformability, so lambda1 (paired with mass1) must be the smaller of
    the two.
    """
    return params[:, param_index["lambda1"]] <= params[:, param_index["lambda2"]]


## For automatically adding all named constraints

_current_module = sys.modules[__name__]

_NAMED_CONSTRAINTS = [
    name
    for name, obj in inspect.getmembers(_current_module, inspect.isfunction)
    if obj.__module__ == __name__ and not name.startswith("_")
]
