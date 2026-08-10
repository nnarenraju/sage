#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : gwosc.py
Description   : Ingest the open-data event portal.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The cumulative endpoint already resolves event versions and supersessions, so it is the
correct source for a baseline list. Concatenating per-release endpoints instead
double-counts events that were re-analysed and re-published under a later release.

Each event carries per-pipeline search entries alongside its parameter-estimation
entries. Those are kept separate: pipelines disagree, and collapsing them to one number
discards the disagreement that a comparison is meant to show.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from sage.search.catalogue.record import CatalogueEvent, Conventions, ExternalCatalogue

CUMULATIVE_ENDPOINT = "https://gwosc.org/eventapi/json/GWTC/"


@dataclass
class PipelineResult:
    """One pipeline's assessment of one event."""

    pipeline: str
    far_per_yr: Optional[float]
    p_astro: Optional[float]
    network_snr: Optional[float]


def load_cumulative(cache, endpoint: str = CUMULATIVE_ENDPOINT) -> ExternalCatalogue:
    """Fetch the version-resolved cumulative event list."""
    raise NotImplementedError


def load_release(cache, release: str) -> ExternalCatalogue:
    """Fetch one named release."""
    raise NotImplementedError


def load_marginal(cache, releases: Optional[Sequence[str]] = None) -> ExternalCatalogue:
    """
    Fetch the marginal and auxiliary lists.

    These are candidates that fell below a release's confidence bar, several of which
    have since been recovered by other groups, so they matter when adjudicating whether
    a Sage candidate is genuinely new.
    """
    raise NotImplementedError


def pipeline_results(event_json: dict) -> List[PipelineResult]:
    """Extract the per-pipeline search entries for one event."""
    raise NotImplementedError


def preferred_posterior_url(event_json: dict) -> Optional[str]:
    """
    Locate the current parameter-estimation release for one event.

    Selection follows the portal's own preference flag rather than order, because
    superseded analyses remain listed alongside the current one.
    """
    raise NotImplementedError
