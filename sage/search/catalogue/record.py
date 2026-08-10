#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : record.py
Description   : The internal catalogue schema.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Every source is reduced to the same record so that comparison logic is written once.
Names are labels only; identity is carried by GPS time.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Conventions:
    """How a source defines the quantities it publishes."""

    significance: str = "far_per_yr"
    mass_frame: str = "source"
    masses_are_template: bool = False
    pastro_prior: str = ""
    detector_networks: Tuple[str, ...] = ()
    searched_mass_range: Optional[Tuple[float, float]] = None
    notes: str = ""

    def significance_comparable_to(self, other: "Conventions") -> bool:
        """Whether two sources' significance values may be compared directly."""
        raise NotImplementedError


@dataclass
class CatalogueEvent:
    """One published event."""

    name: str
    gps: float
    source: str
    far_per_yr: Optional[float] = None
    ifar_yr: Optional[float] = None
    p_astro: Optional[float] = None
    network_snr: Optional[float] = None
    mass1: Optional[float] = None
    mass2: Optional[float] = None
    chirp_mass: Optional[float] = None
    redshift: Optional[float] = None
    luminosity_distance: Optional[float] = None
    chi_eff: Optional[float] = None
    posterior_url: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class ExternalCatalogue:
    """A catalogue and the conventions under which it was produced."""

    key: str
    events: Sequence[CatalogueEvent]
    conventions: Conventions
    reference: str = ""
    version: str = ""
    retrieved_utc: str = ""

    def __len__(self) -> int:
        """Number of events."""
        raise NotImplementedError

    def gps(self) -> np.ndarray:
        """Event times, for matching."""
        raise NotImplementedError

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Columnar view for table building."""
        raise NotImplementedError

    def filter_bbh(self, min_secondary_mass: float = 3.0) -> "ExternalCatalogue":
        """
        Restrict to binary black holes.

        The cut is on the credible lower bound of the secondary mass where a posterior
        is available, so an event is excluded only when it is confidently not a binary
        black hole.
        """
        raise NotImplementedError
