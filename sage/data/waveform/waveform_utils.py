#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : utils.py
Description     : Short description of the file

Created on 2026-03-02 10:20:35

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
import numpy as np


def get_freqs(
    f_l=20.0,
    f_u=1024.0,
    sample_length_in_s=16.0,
    batch_size=None,
    device="cpu",
    dtype=torch.float64,
):
    """
    Build a uniform frequency grid from ``f_l`` to ``f_u`` inclusive.

    The bin spacing is ``df = 1 / sample_length_in_s``, matching the DFT
    convention used throughout Sage.  Optionally broadcasts the grid to a
    batch dimension for vectorised waveform generation.

    Parameters
    ----------
    f_l : float
        Lower frequency bound in Hz (default ``20.0``).
    f_u : float
        Upper frequency bound in Hz (default ``1024.0``).
    sample_length_in_s : float
        Segment duration in seconds; determines ``df = 1/T`` (default ``16.0``).
    batch_size : int or None
        If given, expand the output to shape ``(batch_size, F)``; otherwise
        returns shape ``(F,)`` (default ``None``).
    device : str or torch.device
        Target device (default ``"cpu"``).
    dtype : torch.dtype
        Output dtype (default :attr:`torch.float64`).

    Returns
    -------
    f : torch.Tensor
        Frequency array, shape ``(F,)`` or ``(batch_size, F)``.
    f_ref : torch.Tensor
        Reference frequency (``f_l``), shape ``()`` or ``(batch_size, 1)``.
    """
    # Generate the frequency grid
    del_f = 1.0 / sample_length_in_s
    n = int(np.round((f_u - f_l) / del_f)) + 1
    # This way it will include both f_l and f_u
    f = f_l + del_f * torch.arange(n, device=device, dtype=dtype)
    # Assuming f_ref is f_l
    f_ref = torch.tensor(f_l, device=device, dtype=dtype)

    # Batchify
    if batch_size is not None:
        f = f.unsqueeze(0).expand(batch_size, -1).clone()
        f_ref = f_ref.unsqueeze(0).expand(batch_size, -1).clone()

    return f, f_ref
