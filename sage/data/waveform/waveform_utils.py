#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : utils.py
Description     : Short description of the file

Created on 2026-03-02 10:20:35

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


# Column layout of every waveform parameter batch, in PyCBC names.
WAVEFORM_PARAMETERS = (
    "mass1",
    "mass2",
    "spin1x",
    "spin1y",
    "spin1z",
    "spin2x",
    "spin2y",
    "spin2z",
    "distance",
    "tc",
    "coa_phase",
    "inclination",
    "lambda1",
    "lambda2",
    "polarization",
    "ra",
    "dec",
)
# Parameters that take the LAL default of zero when the prior lacks them.
OPTIONAL_PARAMETERS = ("spin1x", "spin1y", "spin2x", "spin2y", "lambda1", "lambda2")
TRANSVERSE_SPINS = [WAVEFORM_PARAMETERS.index(p) for p in OPTIONAL_PARAMETERS[:4]]
# Columns of the aligned-spin internal layout and its tidal extension.
ALIGNED_COLUMNS = [
    WAVEFORM_PARAMETERS.index(p)
    for p in (
        "mass1",
        "mass2",
        "spin1z",
        "spin2z",
        "distance",
        "tc",
        "coa_phase",
        "inclination",
    )
]
TIDAL_COLUMNS = ALIGNED_COLUMNS + [
    WAVEFORM_PARAMETERS.index(p) for p in ("lambda1", "lambda2")
]
_OPTIONAL = torch.tensor([p in OPTIONAL_PARAMETERS for p in WAVEFORM_PARAMETERS])


def layout_columns(param_index: dict) -> tuple[list[int], list[int]]:
    """
    Sampler and layout columns of the waveform parameters a prior holds.

    Parameters
    ----------
    param_index : dict
        Sampler parameter name to column.

    Returns
    -------
    source, target : list of int
        Sampler columns and their waveform-layout columns.

    Raises
    ------
    ValueError
        If the prior lacks a parameter without a LAL default.
    """
    missing = [
        p
        for p in WAVEFORM_PARAMETERS
        if p not in param_index and p not in OPTIONAL_PARAMETERS
    ]
    if missing:
        raise ValueError(f"prior lacks waveform parameters {missing}")

    present = [i for i, p in enumerate(WAVEFORM_PARAMETERS) if p in param_index]
    return [param_index[WAVEFORM_PARAMETERS[i]] for i in present], present


def to_layout(all_theta: torch.Tensor, source: list, target: list) -> torch.Tensor:
    """
    Waveform-layout batch from a sampler batch; absent parameters are NaN.

    Parameters
    ----------
    all_theta : torch.Tensor, shape (B, P)
        Sampler output.
    source, target : list of int
        Columns from :func:`layout_columns`.

    Returns
    -------
    torch.Tensor, shape (B, 17)
    """
    theta = all_theta.new_full((len(all_theta), len(WAVEFORM_PARAMETERS)), torch.nan)
    theta[:, target] = all_theta[:, source]
    return theta


def fill_defaults(theta: torch.Tensor) -> torch.Tensor:
    """
    float64 copy of a waveform-layout batch, absent optional values zero.

    Parameters
    ----------
    theta : torch.Tensor, shape (B, 17)

    Returns
    -------
    torch.Tensor, shape (B, 17), float64
    """
    theta = theta.to(torch.float64)
    return torch.where(_OPTIONAL.to(theta.device) & theta.isnan(), 0.0, theta)


def aligned_layout(theta: torch.Tensor, columns: list) -> torch.Tensor:
    """
    Internal layout of a non-precessing approximant, checked as LAL does.

    Parameters
    ----------
    theta : torch.Tensor, shape (B, 17)
    columns : list of int
        :data:`ALIGNED_COLUMNS` or :data:`TIDAL_COLUMNS`.

    Returns
    -------
    torch.Tensor, shape (B, len(columns)), float64

    Raises
    ------
    ValueError
        If any transverse spin is non-zero
        (XLALSimInspiralChooseFDWaveform).
    """
    theta = fill_defaults(theta)
    if (theta[:, TRANSVERSE_SPINS] != 0).any():
        raise ValueError(
            "Non-zero transverse spins were given, but this is a "
            "non-precessing approximant."
        )
    return theta[:, columns]
