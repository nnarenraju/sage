#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : test_compiled.py
Description     : Short description of the file

Created on 2026-03-19 11:14:02

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


def test_scatter_equivalence(B=32, S=8, C=4, device="cuda"):

    noise_data = torch.randn(B, C, device=device)
    signal_data = torch.randn(S, C, device=device)

    idx = torch.randperm(B, device=device)[:S]

    # Reference (assignment)
    ref = torch.zeros_like(noise_data)
    ref[idx] = signal_data

    # Scatter version
    scatter = torch.zeros_like(noise_data)
    scatter_idx = idx.view(-1, 1).expand(-1, C)
    scatter = scatter.scatter_add(0, scatter_idx, signal_data)

    return torch.allclose(ref, scatter)
