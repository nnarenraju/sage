#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : subthreshold.py
Description   : Ingest published sub-threshold search data products.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The released candidate lists extend well below the confident threshold, which is where a
new Sage candidate is most likely to have a counterpart. A quiet counterpart in another
pipeline is evidence; its absence at that depth is also informative.

Two details govern correctness. Event times are stored as separate integer second and
nanosecond fields and must be combined arithmetically, not by string concatenation, which
misplaces a trigger by up to a second when the sub-second field is not zero-padded. And
the summary table reports only the single most significant pipeline per event, so a
question about any pipeline must be answered from the per-pipeline tables.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from sage.search.catalogue.record import ExternalCatalogue


@dataclass
class SearchDataProduct:
    """One release of sub-threshold candidates and per-pipeline triggers."""

    root: Path
    release: str
    version_hash: str

    def summary(self) -> Dict[str, np.ndarray]:
        """The per-event summary table."""
        raise NotImplementedError

    def pipeline_table(self, pipeline: str) -> Dict[str, np.ndarray]:
        """One pipeline's own candidate table."""
        raise NotImplementedError

    def pipelines(self) -> Sequence[str]:
        """Pipelines present in the release."""
        raise NotImplementedError

    def any_pipeline_below(self, p_terrestrial: float = 0.5) -> np.ndarray:
        """
        Events any pipeline ranks above a confidence level.

        Formed as a union over the per-pipeline tables, since the summary table carries
        only the most significant pipeline's value for each event.
        """
        raise NotImplementedError


def load(root: str | Path, release: str, version_hash: str) -> SearchDataProduct:
    """Open a staged release."""
    raise NotImplementedError


def read_trigger_xml(path: str | Path, table: str = "coinc_inspiral") -> Dict[str, np.ndarray]:
    """Read a per-event trigger table."""
    raise NotImplementedError


def combine_gps(seconds: np.ndarray, nanoseconds: np.ndarray) -> np.ndarray:
    """Combine integer second and nanosecond fields into a GPS time."""
    raise NotImplementedError


def to_catalogue(product: SearchDataProduct, far_max_per_day: float = 2.0) -> ExternalCatalogue:
    """Convert a release into the internal catalogue schema."""
    raise NotImplementedError
