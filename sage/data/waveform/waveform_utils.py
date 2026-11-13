#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Frequency grids and the shared waveform-parameter layout, with LAL's
refusal of parameters an approximant does not take.

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
    Uniform frequency grid from ``f_l`` to ``f_u`` inclusive.

    The spacing is ``1 / sample_length_in_s``.

    Parameters
    ----------
    f_l : float
        Lower frequency in Hz.
    f_u : float
        Upper frequency in Hz.
    sample_length_in_s : float
        Segment duration in seconds.
    batch_size : int or None
        If given, the grid is repeated to shape ``(batch_size, F)``.
    device : str or torch.device
        Device of the grid.
    dtype : torch.dtype
        Dtype of the grid.

    Returns
    -------
    torch.Tensor
        Frequency array, shape ``(F,)`` or ``(batch_size, F)``.
    """
    del_f = 1.0 / sample_length_in_s
    n = int(np.round((f_u - f_l) / del_f)) + 1
    # n bins include both f_l and f_u
    f = f_l + del_f * torch.arange(n, device=device, dtype=dtype)

    if batch_size is not None:
        f = f.unsqueeze(0).expand(batch_size, -1).clone()

    return f


def reference_frequencies(f_ref, batch_size, device):
    """
    Reference frequency for every row of a batch.

    Parameters
    ----------
    f_ref : float
        Reference frequency in Hz.
    batch_size : int
        Rows in the batch.
    device : str or torch.device
        Device of the result.

    Returns
    -------
    torch.Tensor, shape ``(batch_size, 1)``, float64
        ``f_ref`` repeated.
    """
    return torch.full((batch_size, 1), f_ref, device=device, dtype=torch.float64)


# Column layout of every waveform parameter batch, in PyCBC names;
# delta_tc is the merger's offset (s) from the start of the window
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
    "delta_tc",
    "coa_phase",
    "inclination",
    "lambda1",
    "lambda2",
    "polarization",
    "ra",
    "dec",
)
# Waveform parameters every model takes, and each family's
COMMON_PARAMETERS = ("mass1", "mass2", "distance", "delta_tc", "coa_phase", "inclination", "polarization", "ra", "dec")
TRANSVERSE_SPIN_PARAMETERS = ("spin1x", "spin1y", "spin2x", "spin2y")
TIDAL_PARAMETERS = ("lambda1", "lambda2")
ALIGNED_PARAMETERS = COMMON_PARAMETERS + ("spin1z", "spin2z")
PRECESSING_PARAMETERS = ALIGNED_PARAMETERS + TRANSVERSE_SPIN_PARAMETERS
TIDAL_ALIGNED_PARAMETERS = ALIGNED_PARAMETERS + TIDAL_PARAMETERS
# XLALSimInspiralChooseFDWaveform's refusals of non-zero parameters
# an approximant does not take
REFUSALS = {
    TRANSVERSE_SPIN_PARAMETERS: "Non-zero transverse spins were given, but this is a non-precessing approximant.",
    TIDAL_PARAMETERS: "Non-zero tidal parameters were given, but this is approximant doe not have tidal corrections.",
}
# Columns of the aligned-spin internal layout and its tidal extension
ALIGNED_COLUMNS = [
    WAVEFORM_PARAMETERS.index(p)
    for p in (
        "mass1",
        "mass2",
        "spin1z",
        "spin2z",
        "distance",
        "delta_tc",
        "coa_phase",
        "inclination",
    )
]
TIDAL_COLUMNS = ALIGNED_COLUMNS + [
    WAVEFORM_PARAMETERS.index(p) for p in ("lambda1", "lambda2")
]


def layout_columns(param_index: dict, parameters: tuple, model: str) -> tuple[list[int], list[int]]:
    """
    Sampler and layout columns of the waveform parameters a prior holds.

    Parameters
    ----------
    param_index : dict
        Sampler parameter name to column.
    parameters : tuple of str
        Waveform parameters the model takes.
    model : str
        Model name, for the error.

    Returns
    -------
    source, target : list of int
        Sampler columns and their waveform-layout columns.

    Raises
    ------
    ValueError
        If the prior lacks a parameter the model takes.
    """
    missing = [p for p in parameters if p not in param_index]
    if missing:
        raise ValueError(f"{model} takes the waveform parameters {missing}, which the prior lacks")

    present = [i for i, p in enumerate(WAVEFORM_PARAMETERS) if p in param_index]
    return [param_index[WAVEFORM_PARAMETERS[i]] for i in present], present


def given_extras(param_index: dict, parameters: tuple) -> tuple:
    """
    Waveform parameters the prior gives that the model does not take.

    Parameters
    ----------
    param_index : dict
        Sampler parameter name to column.
    parameters : tuple of str
        Waveform parameters the model takes.

    Returns
    -------
    tuple of str
        The extra parameters, in layout order.
    """
    return tuple(p for p in WAVEFORM_PARAMETERS if p in param_index and p not in parameters)


def refuse_nonzero(theta: torch.Tensor, extras: tuple) -> None:
    """
    Raise as XLALSimInspiralChooseFDWaveform on a non-zero extra.

    An extra is a parameter the approximant does not take; an absent
    value (NaN) counts as not given.

    Parameters
    ----------
    theta : torch.Tensor, shape (B, 17)
        Waveform-layout batch.
    extras : tuple of str
        Parameters the approximant does not take.

    Raises
    ------
    ValueError
        With LAL's message, if any of them is non-zero.
    """
    for group, message in REFUSALS.items():
        names = [p for p in extras if p in group]
        if names:
            values = theta[:, [WAVEFORM_PARAMETERS.index(p) for p in names]]
            if (values.nan_to_num(0.0) != 0).any():
                raise ValueError(message)


def to_layout(all_theta: torch.Tensor, source: list, target: list) -> torch.Tensor:
    """
    Waveform-layout batch from a sampler batch; absent values are NaN.

    Parameters
    ----------
    all_theta : torch.Tensor, shape (B, P)
        Sampler output.
    source, target : list of int
        Columns from :func:`layout_columns`.

    Returns
    -------
    torch.Tensor, shape (B, 17)
        Waveform-layout batch.
    """
    theta = all_theta.new_full((len(all_theta), len(WAVEFORM_PARAMETERS)), torch.nan)
    theta[:, target] = all_theta[:, source]
    return theta


def aligned_layout(theta: torch.Tensor, columns: list, extras: tuple) -> torch.Tensor:
    """
    Internal layout of a non-precessing approximant, as LAL checks it.

    Parameters
    ----------
    theta : torch.Tensor, shape (B, 17)
        Waveform-layout batch.
    columns : list of int
        :data:`ALIGNED_COLUMNS` or :data:`TIDAL_COLUMNS`.
    extras : tuple of str
        Parameters the approximant does not take.

    Returns
    -------
    torch.Tensor, shape (B, len(columns)), float64
        The selected columns.

    Raises
    ------
    ValueError
        With LAL's message, if an extra parameter is non-zero.
    """
    theta = theta.to(torch.float64)
    refuse_nonzero(theta, extras)
    return theta[:, columns]


def precessing_layout(theta: torch.Tensor, extras: tuple) -> torch.Tensor:
    """
    Internal layout of a precessing approximant, checked as LAL does.

    The masses, spins, distance, delta_tc, coa_phase and inclination.

    Parameters
    ----------
    theta : torch.Tensor, shape (B, 17)
        Waveform-layout batch.
    extras : tuple of str
        Parameters the approximant does not take.

    Returns
    -------
    torch.Tensor, shape (B, 12), float64
        The first 12 layout columns.

    Raises
    ------
    ValueError
        With LAL's message, if an extra parameter is non-zero.
    """
    theta = theta.to(torch.float64)
    refuse_nonzero(theta, extras)
    return theta[:, :12]
