#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : _params.py
Description     : Which source parameters are worth sweeping over.

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__maintainer__    = Narenraju Nagarajan


``signal_params`` carries 25 columns, but the per-parameter sweep plots
(efficiency, learning-prior, detected-fraction) are only informative for the
subset the ranking statistic can plausibly depend on.

Sky position, phase and polarisation are drawn isotropically and the detector
response is marginalised over them, so the ranking statistic is flat in those
by construction. Plotting them produced a flat line with autoscaled axes, which
reads as structure when it is scatter -- and 25 figures per sweep per epoch
buried the handful that matter.

They remain available: pass ``params=...`` explicitly, or
``params=ALL_PARAMS`` for the previous behaviour. A flat curve is still a
useful check occasionally; it just should not be the default output.
"""

from __future__ import annotations

__all__ = ["INFORMATIVE_PARAMS", "NUISANCE_PARAMS", "ALL_PARAMS", "select_params"]

#: Parameters the ranking statistic is genuinely expected to depend on.
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
    "tc",
)

#: Isotropic or bookkeeping parameters: flat by construction.
NUISANCE_PARAMS = (
    "ra",
    "dec",
    "coa_phase",
    "polarization",
    "injection_time",
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
    """Filter a ``source_params`` mapping down to the parameters to plot.

    Parameters
    ----------
    source_params : dict[str, array-like]
        Full parameter mapping, as produced by
        :class:`~sage.plotting.manager.ValidationPlotManager`.
    params : iterable[str] or None
        Names to keep. ``None`` keeps everything (:data:`ALL_PARAMS`); pass
        :data:`INFORMATIVE_PARAMS` to drop the isotropic ones. Names that
        are absent from ``source_params`` are skipped, so the same default
        works for BBH and BNS runs with different columns.

    Returns
    -------
    dict[str, array-like]
        Ordered as requested, containing only keys that exist.
    """
    if params is None:
        params = ALL_PARAMS
    return {k: source_params[k] for k in params if k in source_params}
