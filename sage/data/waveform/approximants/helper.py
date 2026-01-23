#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : helper.py
Description     : Short description of the file

Created on 2026-01-23 03:24:33

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


def nudge_backward_(foo: torch.Tensor, max_limit: float, nudge_factor=1e-6) -> None:
    """
    In-place nudge to keep foo <= max_limit with tiny safety margin.
    Modifies foo directly.
    """
    foo.clamp_(max=max_limit - nudge_factor)


def nudge_forward_(foo: torch.Tensor, min_limit: float, nudge_factor=1e-6) -> None:
    """
    In-place nudge to keep foo >= min_limit with tiny safety margin.
    Modifies foo directly.
    """
    foo.clamp_(min=min_limit + nudge_factor)
