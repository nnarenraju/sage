#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : ingest.py
Description   : Loader for the published injection HDF5 sets.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Field names differ between releases, so access goes through an alias table and every
required quantity is validated on load rather than assumed present.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class LVKInjectionSet:
    """Injections, their draw probabilities and the release's own accounting."""

    columns: Dict[str, np.ndarray]
    attrs: Dict[str, object]
    path: Path

    def __len__(self) -> int:
        """Number of injections retained in the file."""
        raise NotImplementedError

    def __getitem__(self, name: str) -> np.ndarray:
        """Column access through the alias table."""
        raise NotImplementedError

    @property
    def total_generated(self) -> int:
        """Injections drawn before the pre-selection cut, needed by the VT estimator."""
        raise NotImplementedError

    def analysis_time_s(self, observing_run: Optional[str] = None) -> float:
        """
        Wall-clock analysis time for one observing run.

        Derived from the injections' own time span and asserted against the release
        attribute, so a combined multi-run release cannot silently supply the wrong
        duration to a single-run search.
        """
        raise NotImplementedError

    def restrict_to_run(self, observing_run: str) -> "LVKInjectionSet":
        """Subset to the injections belonging to one observing run."""
        raise NotImplementedError

    def pipeline_columns(self) -> Tuple[str, ...]:
        """Names of the per-pipeline significance columns present in the release."""
        raise NotImplementedError


def load(path: str | Path, dataset: Optional[str] = None) -> LVKInjectionSet:
    """Read an injection release and validate its required fields."""
    raise NotImplementedError


def spin_convention(injections: LVKInjectionSet) -> str:
    """
    Detect whether spins are stored in cartesian or polar form.

    The two conventions differ by a Jacobian, so a draw probability computed under one
    cannot be combined with a target density expressed in the other.
    """
    raise NotImplementedError
