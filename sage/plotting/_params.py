#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Source parameters drawn by the per-parameter sweep plots, and the
selection of a run's parameters to plot.

Filename        : _params.py
Description     : Which source parameters are worth sweeping over.

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__maintainer__    = Narenraju Nagarajan
"""

from __future__ import annotations

__all__ = ["INFORMATIVE_PARAMS", "NUISANCE_PARAMS", "ALL_PARAMS", "select_params"]

#: Parameters the ranking statistic is expected to depend on.
INFORMATIVE_PARAMS = (
    "mchirp",
    "mass1",
    "mass2",
    "q",
    "distance",
    "chirp_distance",
    "snr",
    "inclination",
    "spin1z",
    "spin2z",
    "delta_tc",
)

#: Isotropic or bookkeeping parameters: flat by construction.
NUISANCE_PARAMS = (
    "ra",
    "dec",
    "coa_phase",
    "polarization",
    "tc",
    "spin1_azimuthal",
    "spin2_azimuthal",
    "spin1_polar",
    "spin2_polar",
    "spin1_a",
    "spin2_a",
    "spin1x",
    "spin1y",
    "spin2x",
    "spin2y",
)

ALL_PARAMS = INFORMATIVE_PARAMS + NUISANCE_PARAMS


def select_params(source_params, params=None):
    """
    The entries of ``source_params`` to plot.

    Parameters
    ----------
    source_params : dict[str, array-like]
        Full parameter mapping, as produced by
        :class:`~sage.plotting.manager.ValidationPlotManager`.
    params : iterable[str] or None
        Names to keep; ``None`` keeps :data:`ALL_PARAMS`. Names absent
        from ``source_params`` are skipped.

    Returns
    -------
    dict[str, array-like]
        The kept entries, in the order of ``params``.
    """
    if params is None:
        params = ALL_PARAMS
    return {k: source_params[k] for k in params if k in source_params}
