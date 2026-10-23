#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : __init__.py
Description     : YAML run specifications for Sage.

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__maintainer__    = Narenraju Nagarajan


A run is described by a short YAML naming a validated ``preset`` plus the few
things that genuinely vary between runs - which observing run's data to use,
on which detectors, which prior ranges, and which stages to execute::

    preset: bbh_production

    data:
      train:    {run: O3a, detectors: [H1, L1]}
      validate: {run: O3b}
      test:     {run: O3b}

    priors:
      masses:   bbh_broad
      spins:    aligned_default
      distance: {min: 100, max: 3000}

    stages: [train, search, benchmark, diagnostics, plots]

Everything else is fixed by the preset. See :mod:`sage.config.schema` for the
full surface and the reasoning behind keeping it small.

This package is additive: it does not change
:func:`sage.core.config.register_configs` or the existing ``runs/*/config.py``
files, which continue to work unchanged.
"""

from sage.config.schema import (
    ConfigError,
    CustomSection,
    DataSection,
    DataSelection,
    KNOWN_STAGES,
    PriorsSection,
    RunSpec,
)
from sage.config.loader import load_run_spec, loads_run_spec, resolve_export_dir

__all__ = [
    "ConfigError",
    "CustomSection",
    "DataSection",
    "DataSelection",
    "KNOWN_STAGES",
    "PriorsSection",
    "RunSpec",
    "load_run_spec",
    "loads_run_spec",
    "resolve_export_dir",
]
