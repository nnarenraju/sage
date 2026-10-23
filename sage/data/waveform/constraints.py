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
    """
    Enforce mass1 >= mass2 in a torch.compile-safe way.

    Reorders masses per sample using pure tensor ops.
    """

    idx_m1 = param_index["mass1"]
    idx_m2 = param_index["mass2"]

    m1 = params[:, idx_m1]
    m2 = params[:, idx_m2]

    # Compute ordered masses (no branching)
    m1_new = torch.maximum(m1, m2)
    m2_new = torch.minimum(m1, m2)

    # Write back
    params[:, idx_m1] = m1_new
    params[:, idx_m2] = m2_new

    return params


def lambda_order(params, param_index, extra_params=None):
    """
    Enforce lambda1 <= lambda2 in a torch.compile-safe way.

    Astrophysically the more massive component (mass1, after
    :func:`mass_order`) is more compact and has the SMALLER tidal
    deformability, so lambda1 (paired with mass1) must be the smaller of
    the two.  Reorders the per-star tidal deformabilities using pure
    tensor ops, mirroring :func:`mass_order`.
    """

    idx_l1 = param_index["lambda1"]
    idx_l2 = param_index["lambda2"]

    l1 = params[:, idx_l1]
    l2 = params[:, idx_l2]

    # lambda1 (heavier star) <= lambda2 (lighter star), no branching
    params[:, idx_l1] = torch.minimum(l1, l2)
    params[:, idx_l2] = torch.maximum(l1, l2)

    return params


## For automatically adding all named constraints

_current_module = sys.modules[__name__]

_NAMED_CONSTRAINTS = [
    name
    for name, obj in inspect.getmembers(_current_module, inspect.isfunction)
    if obj.__module__ == __name__ and not name.startswith("_")
]
