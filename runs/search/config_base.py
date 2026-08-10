#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : config_base.py
Description   : Shared search-campaign configuration.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

One campaign covers one observing run. Whitening is the network's input normalisation, so
the fiducial spectra must belong to the run being searched; background, false-alarm rates
and the noise density are likewise per-run. Cross-run figures are assembled afterwards
from the per-run products.

Training configuration comes from the run that produced the checkpoint, imported through
the usual mechanism so that the search reproduces the geometry the network was trained
with. Search-only settings are applied on top without touching the training run.
"""

from pathlib import Path
from typing import Optional, Sequence

SEARCH_ROOT = Path("/work/nagarajan/sage_runs/search")

RELEASE_DIRS = {
    "O3a": Path("/work/nagarajan/data_release_o3a"),
    "O3b": Path("/work/nagarajan/data_release"),
    "O4a": Path("/work/nagarajan/data_release_o4a"),
    "O4b": Path("/work/nagarajan/data_release_o4b"),
}


def make_spec(
    observing_run: str,
    checkpoint: str | Path,
    training_config: str,
    fiducial_dir: str | Path,
    detectors: Sequence[str] = ("H1", "L1"),
    tag: Optional[str] = None,
    n_slides: int = 82,
    **overrides,
):
    """
    Build the search specification for one observing run.

    Parameters
    ----------
    observing_run : str
        Key into the release directories.
    checkpoint : path
        Trained weights; its stored configuration is validated against the live one.
    training_config : str
        Config module of the run that produced the checkpoint.
    fiducial_dir : path
        Fiducial spectra for this observing run.
    n_slides : int
        Time slides for the background. Background livetime is measured from the
        resulting plan, not inferred from this number.
    """
    raise NotImplementedError


def register(spec) -> None:
    """Load the training configuration and apply the search overrides on top of it."""
    raise NotImplementedError
