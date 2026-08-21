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


def parse_allevents(payload: dict, key: str = "gwosc") -> ExternalCatalogue:
    """
    Build an event list from a GWOSC ``allevents`` payload.

    Separated from fetching so the parsing is testable without a network, and so a
    payload obtained any other way -- a cached file, a colleague's copy -- goes through
    exactly the same code as a live fetch.

    ``GPS`` is required per event and everything else is optional. GWOSC publishes
    different fields for different releases, and an event missing a FAR is an event whose
    FAR that release did not report, not a malformed record.
    """
    events = payload.get("events", payload)
    out = []
    for name, record in sorted(events.items()):
        if "GPS" not in record or record["GPS"] is None:
            raise ValueError(
                f"GWOSC event {name!r} carries no GPS time; it cannot be placed in time "
                "and so cannot be compared against anything"
            )
        out.append(
            CatalogueEvent(
                name=str(name),
                gps=float(record["GPS"]),
                source=str(key),
                far_per_yr=_number(record.get("far")),
                p_astro=_number(record.get("p_astro")),
                network_snr=_number(record.get("network_matched_filter_snr")),
                mass1=_number(record.get("mass_1_source")),
                mass2=_number(record.get("mass_2_source")),
                chirp_mass=_number(record.get("chirp_mass_source")),
                redshift=_number(record.get("redshift")),
                luminosity_distance=_number(record.get("luminosity_distance")),
                chi_eff=_number(record.get("chi_eff")),
                extra={
                    "catalog": record.get("catalog.shortName", ""),
                    "version": record.get("version", ""),
                    # The credible lower bound where GWOSC publishes one, which is what
                    # filter_bbh reads: an event is excluded from the BBH list only when
                    # it is confidently not one.
                    "mass2_lower_bound": _credible_bound(
                        record.get("mass_2_source"), record.get("mass_2_source_lower")
                    ),
                },
            )
        )
    return ExternalCatalogue(
        key=str(key),
        events=out,
        conventions=Conventions(
            significance="far_per_yr",
            mass_frame="source",
            masses_are_template=False,
            notes=(
                "GWOSC cumulative event list. FAR is the release's preferred pipeline "
                "value; p_astro where published is under that pipeline's own prior."
            ),
        ),
        reference="https://gwosc.org/eventapi/",
    )


def _credible_bound(estimate, offset):
    """
    The lower end of a GWOSC credible interval, as a mass rather than an error bar.

    ``mass_2_source_lower`` is a **signed offset** from the median, not an absolute
    bound: ``mass_2_source = 18.5`` with ``mass_2_source_lower = -4.0`` means the
    interval reaches down to 14.5. Reading the field directly makes every event with a
    published error bar compare as -4.0 solar masses, so a BBH cut at 3.0 excludes the
    entire confident catalogue and the recovery gate scores a found event as missed.

    Returns ``None`` when either field is absent, which :meth:`filter_bbh` reads as "no
    measurement" and keeps.
    """
    median = _number(estimate)
    delta = _number(offset)
    if median is None or delta is None:
        return median
    bound = median + delta
    if delta > 0.0 or bound <= 0.0:
        # The offset is signed by convention, so a positive one means the convention
        # changed under us. Fall back to the point estimate rather than invent a bound:
        # a wrong bound is silent, and this way the event is judged on a real mass.
        return median
    return bound


def _number(value):
    """A float, or ``None`` for a missing or unparseable field."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def fetch_allevents(cache=None, host: str = "https://gwosc.org") -> dict:
    """
    Fetch the cumulative event list from GWOSC.

    Uses ``gwosc.api.fetch_allevents_json``, which is the same call
    :mod:`sage.data.primer.get_segments` already makes to excise known events from a
    training release -- so the search compares against exactly the list the data
    preparation knew about, rather than a second opinion assembled here.

    The payload is written into ``cache`` when one is given, which is what makes a later
    run reproducible and offline: the comparison then reads frozen bytes rather than
    whatever the service returns that day.
    """
    from gwosc.api import fetch_allevents_json

    payload = fetch_allevents_json(full=True, host=host)
    if cache is not None:
        import json

        cache.put(f"{host}/eventapi/json/allevents/", json.dumps(payload).encode())
    return payload


def load_cumulative(cache, endpoint: str = CUMULATIVE_ENDPOINT) -> ExternalCatalogue:
    """
    The version-resolved cumulative event list, from the cache where possible.

    The cumulative endpoint already resolves event versions and supersessions, so it is
    the correct source for a baseline list; concatenating per-release endpoints instead
    double-counts events that were re-analysed and re-published under a later release.

    Reads the cache first and only reaches the network when the entry is absent. A
    campaign with a frozen manifest therefore never touches the service, which is what
    lets it be re-run years later against the catalogue it actually used.
    """
    import json

    entry = cache.fetch(endpoint)
    return parse_allevents(json.loads(entry.path.read_bytes()), key="gwosc")


def load_release(cache, release: str) -> ExternalCatalogue:
    """
    Fetch one named release, such as ``GWTC-2.1-confident``.

    Kept separate from :func:`load_cumulative` because a release list and the cumulative
    list answer different questions: the first is what one paper claimed, the second is
    what is currently believed.
    """
    import json

    entry = cache.fetch(f"https://gwosc.org/eventapi/json/{release}/")
    return parse_allevents(json.loads(entry.path.read_bytes()), key=release)


def load_marginal(cache, releases: Optional[Sequence[str]] = None) -> ExternalCatalogue:
    """
    Fetch the marginal and auxiliary lists.

    These are candidates that fell below a release's confidence bar, several of which
    have since been recovered by other groups, so they matter when adjudicating whether
    a Sage candidate is genuinely new: a candidate matching a marginal event is not a
    discovery, it is a confirmation.
    """
    releases = tuple(releases or ("GWTC-1-marginal", "GWTC-2.1-marginal"))
    merged: List[CatalogueEvent] = []
    for release in releases:
        merged.extend(load_release(cache, release).events)
    return ExternalCatalogue(
        key="gwosc-marginal",
        events=merged,
        conventions=Conventions(
            significance="far_per_yr",
            notes="Sub-threshold and marginal candidates from the named releases.",
        ),
        reference="https://gwosc.org/eventapi/",
        version=",".join(releases),
    )


def pipeline_results(event_json: dict) -> List[PipelineResult]:
    """
    Extract the per-pipeline search entries for one event.

    Kept separate from the event's own significance because pipelines disagree, and
    collapsing them to one number discards exactly the disagreement a comparison is meant
    to show.
    """
    out = []
    for entry in event_json.get("parameters", {}).values():
        if not isinstance(entry, dict) or not entry.get("is_pipeline", False):
            continue
        out.append(
            PipelineResult(
                pipeline=str(entry.get("pipeline", "") or entry.get("name", "")),
                far_per_yr=_number(entry.get("far")),
                p_astro=_number(entry.get("p_astro")),
                network_snr=_number(entry.get("network_matched_filter_snr")),
            )
        )
    return out


def preferred_posterior_url(event_json: dict) -> Optional[str]:
    """
    Locate the current parameter-estimation release for one event.

    Selection follows the portal's own preference flag rather than order, because
    superseded analyses remain listed alongside the current one and the newest entry is
    not reliably the preferred one.
    """
    for entry in event_json.get("parameters", {}).values():
        if not isinstance(entry, dict):
            continue
        if entry.get("is_preferred", False):
            for field in ("data_url", "url", "posterior_url"):
                if entry.get(field):
                    return str(entry[field])
    return None
