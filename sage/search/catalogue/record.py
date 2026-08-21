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
    #: ``(start, end)`` GPS of the time this source searched, when it states one.
    #: Without it, coverage falls back to the span of the events the catalogue contains,
    #: which is a *lower* bound: a source that published nothing near the start of a run
    #: still searched there. The fallback therefore under-claims coverage, and that is
    #: the safe direction -- under-claimed coverage means under-claimed new events, where
    #: over-claiming it would manufacture discoveries at the edge of somebody else's
    #: scope and report their published events as missed.
    searched_gps_span: Optional[Tuple[float, float]] = None
    notes: str = ""

    def significance_comparable_to(self, other: "Conventions") -> bool:
        """
        Whether two sources' significance values may be compared directly.

        A FAR and a p_astro are not the same quantity and never become one: a rate has
        units and is unbounded, a probability has neither, and no monotone map between
        them exists without both pipelines' rate models. Two p_astro values computed
        under different priors are not comparable either -- the prior is what turns a
        likelihood ratio into a probability, so the same event scores differently under
        each and the difference says nothing about the data.

        Returned rather than raised so a caller can present both values side by side and
        say they are incomparable, which is more useful than refusing to show them.
        """
        if self.significance != other.significance:
            return False
        if self.significance == "p_astro":
            return bool(self.pastro_prior) and self.pastro_prior == other.pastro_prior
        return True


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
        return len(self.events)

    def gps(self) -> np.ndarray:
        """Event times, for matching."""
        return np.asarray([event.gps for event in self.events], dtype=np.float64)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """
        Columnar view for table building.

        A field absent from an event becomes ``nan`` rather than being omitted, so every
        column has one entry per event and a catalogue that publishes chirp mass for some
        events and not others still lines up row for row.
        """
        fields = (
            "far_per_yr", "ifar_yr", "p_astro", "network_snr", "mass1", "mass2",
            "chirp_mass", "redshift", "luminosity_distance", "chi_eff",
        )
        out: Dict[str, np.ndarray] = {
            "name": np.asarray([str(e.name) for e in self.events]),
            "gps": self.gps(),
            "source": np.asarray([str(e.source) for e in self.events]),
        }
        for field_name in fields:
            out[field_name] = np.asarray(
                [
                    np.nan if getattr(e, field_name) is None
                    else float(getattr(e, field_name))
                    for e in self.events
                ],
                dtype=np.float64,
            )
        return out

    def filter_bbh(self, min_secondary_mass: float = 3.0) -> "ExternalCatalogue":
        """
        Restrict to binary black holes.

        The cut is on the credible lower bound of the secondary mass where a posterior
        is available, so an event is excluded only when it is confidently not a binary
        black hole.

        An event with no secondary mass at all is **kept**. Absence of a measurement is
        not evidence of a light companion, and dropping such events would quietly shrink
        the list the search is scored against -- turning a missing column into a missed
        recovery.
        """
        kept = []
        for event in self.events:
            # Explicit, because `extra` carries the key with a None value whenever the
            # source published no lower bound -- and `get(key, default)` returns that
            # stored None rather than the default, so the fallback to the point estimate
            # never happened and every event was kept. GW190425 (m2 = 1.4) then sits in
            # the BBH list, and the recovery gate counts a BNS the search never looked
            # for as a miss.
            bound = event.extra.get("mass2_lower_bound")
            if bound is None:
                bound = event.mass2
            if bound is None or float(bound) >= float(min_secondary_mass):
                kept.append(event)
        return ExternalCatalogue(
            key=self.key,
            events=tuple(kept),
            conventions=self.conventions,
            reference=self.reference,
            version=self.version,
            retrieved_utc=self.retrieved_utc,
        )
