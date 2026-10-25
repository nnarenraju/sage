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
    torch.Tensor
        Frequency array, shape ``(F,)`` or ``(batch_size, F)``.
    """
    # Generate the frequency grid
    del_f = 1.0 / sample_length_in_s
    n = int(np.round((f_u - f_l) / del_f)) + 1
    # This way it will include both f_l and f_u
    f = f_l + del_f * torch.arange(n, device=device, dtype=dtype)

    # Batchify
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
# Waveform parameters every model takes, and each family's.
COMMON_PARAMETERS = ("mass1", "mass2", "distance", "tc", "coa_phase", "inclination", "polarization", "ra", "dec")
TRANSVERSE_SPIN_PARAMETERS = ("spin1x", "spin1y", "spin2x", "spin2y")
TIDAL_PARAMETERS = ("lambda1", "lambda2")
ALIGNED_PARAMETERS = COMMON_PARAMETERS + ("spin1z", "spin2z")
PRECESSING_PARAMETERS = ALIGNED_PARAMETERS + TRANSVERSE_SPIN_PARAMETERS
TIDAL_ALIGNED_PARAMETERS = ALIGNED_PARAMETERS + TIDAL_PARAMETERS
# XLALSimInspiralChooseFDWaveform's refusals of non-zero parameters an approximant does not take.
REFUSALS = {
    TRANSVERSE_SPIN_PARAMETERS: "Non-zero transverse spins were given, but this is a non-precessing approximant.",
    TIDAL_PARAMETERS: "Non-zero tidal parameters were given, but this is approximant doe not have tidal corrections.",
}
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
    """Waveform parameters the prior gives that the model does not take."""
    return tuple(p for p in WAVEFORM_PARAMETERS if p in param_index and p not in parameters)


def refuse_nonzero(theta: torch.Tensor, extras: tuple) -> None:
    """
    Raise as XLALSimInspiralChooseFDWaveform does when a parameter the
    approximant does not take is non-zero; an absent value (NaN) is not given.

    Parameters
    ----------
    theta : torch.Tensor, shape (B, 17)
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


def aligned_layout(theta: torch.Tensor, columns: list, extras: tuple) -> torch.Tensor:
    """
    Internal layout of a non-precessing approximant, checked as LAL does.

    Parameters
    ----------
    theta : torch.Tensor, shape (B, 17)
    columns : list of int
        :data:`ALIGNED_COLUMNS` or :data:`TIDAL_COLUMNS`.
    extras : tuple of str
        Parameters the approximant does not take (:func:`refuse_nonzero`).

    Returns
    -------
    torch.Tensor, shape (B, len(columns)), float64
    """
    theta = theta.to(torch.float64)
    refuse_nonzero(theta, extras)
    return theta[:, columns]


def precessing_layout(theta: torch.Tensor, extras: tuple) -> torch.Tensor:
    """
    Internal layout of a precessing approximant (masses, spins, distance, tc,
    coa_phase, inclination), checked as LAL does.

    Parameters
    ----------
    theta : torch.Tensor, shape (B, 17)
    extras : tuple of str
        Parameters the approximant does not take (:func:`refuse_nonzero`).

    Returns
    -------
    torch.Tensor, shape (B, 12), float64
    """
    theta = theta.to(torch.float64)
    refuse_nonzero(theta, extras)
    return theta[:, :12]
