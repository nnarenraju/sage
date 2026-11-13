#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Named constraints on prior samples, as boolean masks for the
rejection sampling in :mod:`sage.data.waveform.sampler`.

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

import sys
import torch
import inspect


def mass_order(params, param_index, extra_params=None):
    """
    Mask of the samples with ``mass1 >= mass2`` (PyCBC's constraint).

    Parameters
    ----------
    params : torch.Tensor, shape ``(N, num_params)``
        Prior samples.
    param_index : dict[str, int]
        Column of each parameter in ``params``.
    extra_params : list or None
        Unused; part of the constraint signature.

    Returns
    -------
    torch.Tensor, shape ``(N,)``
        True where ``mass1 >= mass2``.
    """
    return params[:, param_index["mass1"]] >= params[:, param_index["mass2"]]


def lambda_order(params, param_index, extra_params=None):
    """
    Mask of the samples with ``lambda1 <= lambda2``.

    The more massive component (mass1, after :func:`mass_order`) is
    the more compact one and has the smaller tidal deformability.

    Parameters
    ----------
    params : torch.Tensor, shape ``(N, num_params)``
        Prior samples.
    param_index : dict[str, int]
        Column of each parameter in ``params``.
    extra_params : list or None
        Unused; part of the constraint signature.

    Returns
    -------
    torch.Tensor, shape ``(N,)``
        True where ``lambda1 <= lambda2``.
    """
    return params[:, param_index["lambda1"]] <= params[:, param_index["lambda2"]]


# Public functions of this module, the constraints selectable by name

_current_module = sys.modules[__name__]

_NAMED_CONSTRAINTS = [
    name
    for name, obj in inspect.getmembers(_current_module, inspect.isfunction)
    if obj.__module__ == __name__ and not name.startswith("_")
]
