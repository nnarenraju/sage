#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : crossmatch.py
Description   : Match candidates against catalogues on GPS time.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Matching is always on GPS time, never on name: the same event is published with
second-level differences in its name between catalogues, so name matching both misses
real associations and invents false ones. Times are compared in integer nanoseconds to
avoid float drift across sources that quote different precision.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MatchResult:
    """Association between two event lists."""

    left_index: np.ndarray
    right_index: np.ndarray
    dt_s: np.ndarray
    unmatched_left: np.ndarray
    unmatched_right: np.ndarray

    def as_dict(self) -> dict:
        """Flat summary for the manifest."""
        return {
            "n_matched": int(self.left_index.size),
            "n_unmatched_left": int(self.unmatched_left.size),
            "n_unmatched_right": int(self.unmatched_right.size),
            "max_dt_s": float(np.max(np.abs(self.dt_s))) if self.dt_s.size else 0.0,
            "median_dt_s": (
                float(np.median(np.abs(self.dt_s))) if self.dt_s.size else 0.0
            ),
        }


#: GPS times are compared in integer nanoseconds. Sources quote times to differing
#: precision -- some to the second, some to six decimals -- and a float comparison of
#: numbers of order 1.2e9 carries about 0.24 microseconds of resolution, which is enough
#: to make a tolerance behave differently depending on which source was read first.
_NS: int = 1_000_000_000


def merger_times(columns) -> np.ndarray:
    """
    The times a candidate is matched to a published event on.

    ``tc_gps`` -- the decoded coalescence time -- not ``gps``, which is where the analysis
    window starts. A catalogue publishes a merger time, so joining on the window start
    compares two different quantities: measured on the O3a smoke campaign the two differ
    by **13.05 s**, twenty times the 1.0 s match tolerance, and every one of the five
    published events the search had actually recovered was reported as new. Against
    ``tc_gps`` the same five match to between 0.009 and 0.094 s.

    ``gps`` is the fallback, for a campaign whose engine carried no decoder and therefore
    estimated no coalescence time. Such a campaign cannot place a candidate to better than
    a window, and matching it on the window start is the honest thing left to do.
    """
    if "tc_gps" in columns:
        return np.asarray(columns["tc_gps"], dtype=np.float64).ravel()
    return np.asarray(columns["gps"], dtype=np.float64).ravel()


def match_on_gps(
    gps_left: np.ndarray,
    gps_right: np.ndarray,
    tolerance_s: float = 1.0,
) -> MatchResult:
    """
    Associate two event lists by nearest GPS time within a tolerance.

    One-to-one. Each match consumes both entries, so two candidates close together
    cannot both claim the same published event -- which would report a recovery twice
    and leave the second candidate looking new.

    Pairs are taken in order of separation, closest first, which makes the assignment
    independent of the order the lists arrive in. A greedy sweep in index order gives a
    different answer depending on that order, and both answers look equally reasonable
    afterwards.

    Matching is on time and never on name: the same event is published with second-level
    differences in its name between catalogues, so name matching both misses real
    associations and invents false ones.
    """
    left = np.asarray(gps_left, dtype=np.float64).ravel()
    right = np.asarray(gps_right, dtype=np.float64).ravel()
    if not np.isfinite(tolerance_s) or tolerance_s <= 0:
        raise ValueError(
            f"tolerance_s must be finite and positive, got {tolerance_s}"
        )
    if left.size == 0 or right.size == 0:
        return MatchResult(
            left_index=np.zeros(0, dtype=np.int64),
            right_index=np.zeros(0, dtype=np.int64),
            dt_s=np.zeros(0, dtype=np.float64),
            unmatched_left=np.arange(left.size, dtype=np.int64),
            unmatched_right=np.arange(right.size, dtype=np.int64),
        )

    left_ns = np.rint(left * _NS).astype(np.int64)
    right_ns = np.rint(right * _NS).astype(np.int64)
    tolerance_ns = int(round(float(tolerance_s) * _NS))

    # Every pair inside the tolerance, then resolve conflicts closest-first. The lists
    # here are catalogues -- hundreds of events, not millions -- so the full pairing is
    # cheap and is what makes the result order-independent.
    separation = np.abs(left_ns[:, None] - right_ns[None, :])
    rows, columns = np.nonzero(separation <= tolerance_ns)
    order = np.argsort(separation[rows, columns], kind="stable")

    taken_left, taken_right = set(), set()
    pairs = []
    for index in order:
        i, j = int(rows[index]), int(columns[index])
        if i in taken_left or j in taken_right:
            continue
        taken_left.add(i)
        taken_right.add(j)
        pairs.append((i, j))
    pairs.sort()

    matched_left = np.asarray([i for i, _ in pairs], dtype=np.int64)
    matched_right = np.asarray([j for _, j in pairs], dtype=np.int64)
    return MatchResult(
        left_index=matched_left,
        right_index=matched_right,
        dt_s=(
            (left_ns[matched_left] - right_ns[matched_right]).astype(np.float64) / _NS
            if pairs
            else np.zeros(0, dtype=np.float64)
        ),
        unmatched_left=np.asarray(
            sorted(set(range(left.size)) - taken_left), dtype=np.int64
        ),
        unmatched_right=np.asarray(
            sorted(set(range(right.size)) - taken_right), dtype=np.int64
        ),
    )


def classify(
    candidates,
    catalogues: Dict[str, object],
    tolerance_s: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Label each candidate against each catalogue, and each catalogue event against us.

    Two directions, because they answer different questions and only one of them is about
    our candidates:

    - **known / new**, per candidate: did any catalogue publish an event at this time?
      A candidate matching none of them is what "new" means, and it is the whole point of
      running the search.
    - **recovered / missed**, per catalogue event: did we produce a candidate at its
      time? This is the validation gate -- a search that misses published events is not
      ready to claim anything about the ones it does find.

    ``covered`` qualifies both. A catalogue that did not search a region says nothing
    about it, so an event absent from one is only *missed* by that catalogue where it
    searched -- and a candidate is only *new* where the catalogues that could have seen
    it did not. Treating absence as a null result outside coverage is how a search
    invents discoveries at the edge of somebody else's parameter space.
    """
    columns = getattr(candidates, "columns", candidates)
    gps = merger_times(columns)
    mchirp = (
        np.asarray(columns["mchirp"], dtype=np.float64).ravel()
        if "mchirp" in columns
        else None
    )

    known = np.zeros(gps.size, dtype=bool)
    matched_name = np.full(gps.size, "", dtype=object)
    matched_dt = np.full(gps.size, np.nan, dtype=np.float64)
    matched_source = np.full(gps.size, "", dtype=object)
    any_coverage = np.zeros(gps.size, dtype=bool)

    out: Dict[str, np.ndarray] = {}
    for key, catalogue in catalogues.items():
        result = match_on_gps(gps, catalogue.gps(), tolerance_s=tolerance_s)
        covered = coverage_mask(catalogue, gps, mchirp)
        any_coverage |= covered

        recovered_here = np.zeros(gps.size, dtype=bool)
        recovered_here[result.left_index] = True
        out[f"{key}_matched"] = recovered_here
        out[f"{key}_covered"] = covered
        out[f"{key}_dt_s"] = _scatter(gps.size, result.left_index, result.dt_s)

        names = np.asarray([str(e.name) for e in catalogue.events])
        event_recovered = np.zeros(names.size, dtype=bool)
        event_recovered[result.right_index] = True
        out[f"{key}_event_name"] = names
        out[f"{key}_event_recovered"] = event_recovered

        first = recovered_here & ~known
        matched_name[first] = names[result.right_index][
            np.isin(result.left_index, np.flatnonzero(first))
        ]
        matched_source[first] = str(key)
        matched_dt[first] = out[f"{key}_dt_s"][first]
        known |= recovered_here

    out["known"] = known
    # New only where something could have seen it. Elsewhere the question was not asked.
    out["is_new"] = (~known) & any_coverage
    out["covered_by_any"] = any_coverage
    out["catalogue_match"] = np.asarray([str(v) for v in matched_name])
    out["catalogue_source"] = np.asarray([str(v) for v in matched_source])
    out["catalogue_dt_s"] = matched_dt
    return out


def _scatter(size: int, index: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Place ``values`` at ``index`` in a nan-filled array of length ``size``."""
    out = np.full(int(size), np.nan, dtype=np.float64)
    if index.size:
        out[index] = values
    return out


def overlap_sets(
    catalogues: Dict[str, np.ndarray], tolerance_s: float = 1.0
) -> Dict[Tuple[str, ...], List[int]]:
    """
    Resolve events shared between catalogues into disjoint membership sets.

    Keyed by the sorted tuple of catalogues an event appears in, so every event is
    counted once and the sets partition the union -- which is what makes the counts add
    up to the number of distinct events rather than to the number of publications.

    Events are grouped transitively: A within tolerance of B and B of C puts all three
    together even when A and C are further apart than the tolerance. Two catalogues
    quoting a time to different precision are exactly this case, and splitting them would
    report one event as two.

    Values are indices into the *union* built here, returned alongside it by
    :func:`union_times`, since an index into any one catalogue would not identify an
    event shared between several.
    """
    keys = sorted(catalogues)
    entries = []
    for key in keys:
        for time in np.asarray(catalogues[key], dtype=np.float64).ravel():
            entries.append((float(time), key))
    if not entries:
        return {}
    entries.sort()

    tolerance = float(tolerance_s)
    groups: List[dict] = []
    for time, key in entries:
        # A group never takes two entries from the same source. Within tolerance of each
        # other but published by one catalogue, they are two events by that catalogue's
        # own reckoning -- merging them would overrule the source about its own list.
        #
        # This is not hypothetical: the crossmatch tolerance is wider than the clustering
        # window that separates our own candidates (1.0 s against 0.35 s), so without
        # this rule two candidates our clustering had already ruled distinct are counted
        # as one event.
        joins = (
            groups
            and time - groups[-1]["last"] <= tolerance
            and key not in groups[-1]["members"]
        )
        if joins:
            groups[-1]["members"].add(key)
            groups[-1]["last"] = time
            groups[-1]["times"].append(time)
        else:
            groups.append({"members": {key}, "last": time, "times": [time]})

    out: Dict[Tuple[str, ...], List[int]] = {}
    for index, group in enumerate(groups):
        out.setdefault(tuple(sorted(group["members"])), []).append(index)
    return out


def union_times(
    catalogues: Dict[str, np.ndarray], tolerance_s: float = 1.0
) -> np.ndarray:
    """
    Representative time of each distinct event across the catalogues.

    The median of the grouped times, not the first: sources quote a time to differing
    precision and the first one encountered is an accident of sort order, whereas the
    median is a property of the group.
    """
    keys = sorted(catalogues)
    entries = sorted(
        (float(t), k)
        for k in keys
        for t in np.asarray(catalogues[k], dtype=np.float64).ravel()
    )
    if not entries:
        return np.zeros(0, dtype=np.float64)
    tolerance = float(tolerance_s)
    groups: List[List[float]] = []
    sources: List[set] = []
    last = None
    for time, key in entries:
        joins = (
            last is not None
            and time - last <= tolerance
            and key not in sources[-1]
        )
        if joins:
            groups[-1].append(time)
            sources[-1].add(key)
        else:
            groups.append([time])
            sources.append({key})
        last = time
    return np.asarray([float(np.median(g)) for g in groups], dtype=np.float64)


def comparison_table(
    candidates,
    catalogues: Dict[str, object],
    tolerance_s: float = 1.0,
) -> dict:
    """
    Build the wide event-by-catalogue comparison.

    Rows are events, columns are catalogues, cells carry each catalogue's significance,
    and entries unique to one catalogue are flagged.

    **The significance columns are not placed on one axis.** Each carries the quantity
    that catalogue publishes, under its own conventions, and ``comparable`` records
    which of them may be read against this search's. A FAR and a p_astro are different
    quantities; two p_astro values under different priors are different quantities too.
    Putting them in adjacent columns is presentation; asserting they are the same number
    would be a claim, and this refuses to make it.
    """
    columns = getattr(candidates, "columns", candidates)
    gps = merger_times(columns)

    times = {key: cat.gps() for key, cat in catalogues.items()}
    times["_sage"] = gps
    events = union_times(times, tolerance_s=tolerance_s)
    membership = overlap_sets(times, tolerance_s=tolerance_s)
    owners = np.full(events.size, None, dtype=object)
    for members, indices in membership.items():
        for index in indices:
            owners[index] = members

    table: Dict[str, np.ndarray] = {
        "gps": events,
        "n_sources": np.asarray(
            [0 if owners[i] is None else len(owners[i]) for i in range(events.size)],
            dtype=np.int64,
        ),
        "in_sage": np.asarray(
            [bool(owners[i]) and "_sage" in owners[i] for i in range(events.size)]
        ),
    }
    conventions = {}
    for key, catalogue in catalogues.items():
        arrays = catalogue.to_arrays()
        result = match_on_gps(events, catalogue.gps(), tolerance_s=tolerance_s)
        present = np.zeros(events.size, dtype=bool)
        present[result.left_index] = True
        table[f"{key}_present"] = present
        significance = catalogue.conventions.significance
        values = np.full(events.size, np.nan, dtype=np.float64)
        if significance in arrays:
            values[result.left_index] = arrays[significance][result.right_index]
        table[f"{key}_{significance}"] = values
        conventions[key] = significance

    table["unique_to_one"] = table["n_sources"] == 1
    return {
        "table": table,
        "significance_by_source": conventions,
        "n_events": int(events.size),
        "tolerance_s": float(tolerance_s),
    }


def coverage_mask(catalogue, gps: np.ndarray, mchirp: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Whether each candidate falls inside a catalogue's searched parameter space and time.

    Absence from a catalogue carries no information where that catalogue did not search,
    so coverage is recorded per catalogue instead of treating absence as a null result.

    Time coverage is bounded by the catalogue's own event span when it declares nothing
    better. That is deliberately weak -- it is a lower bound on where the catalogue
    looked, not a statement of its segments -- and it errs toward *not* claiming coverage,
    which is the direction that avoids inventing a missed event.

    The mass cut uses ``searched_mass_range`` when the source declares one. Without a
    chirp mass for the candidate the parameter-space question cannot be asked, and the
    answer is time coverage alone rather than a guess.
    """
    gps = np.asarray(gps, dtype=np.float64).ravel()
    times = catalogue.gps()
    if times.size == 0:
        return np.zeros(gps.size, dtype=bool)

    span = getattr(catalogue.conventions, "searched_gps_span", None)
    if span is not None:
        lo, hi = float(span[0]), float(span[1])
    else:
        lo, hi = float(times.min()), float(times.max())
    covered = (gps >= lo) & (gps <= hi)

    mass_range = getattr(catalogue.conventions, "searched_mass_range", None)
    if mass_range is not None and mchirp is not None:
        mchirp = np.asarray(mchirp, dtype=np.float64).ravel()
        covered &= (mchirp >= float(mass_range[0])) & (mchirp <= float(mass_range[1]))
    return covered


def _analysed_intervals(spec) -> np.ndarray:
    """
    The time this campaign analysed, as ``(n, 2)`` GPS intervals.

    Read from the window lattice rather than from the observing segments: a window needs
    a whole window of contiguous data in every detector, so analysed time is strictly
    less than coincident time, and the larger number would claim the search could have
    seen an event at a moment it could not have triggered on.
    """
    from sage.search.trials import analysed_intervals

    intervals, _ = analysed_intervals(spec)
    return intervals


def _restrict(catalogue, intervals: np.ndarray):
    """
    Keep only the events inside the analysed time, recording the span on the result.

    Restriction is what makes "missed" mean something. An event outside the analysed time
    was never searched for, and a recovery report that lists it says the search failed at
    something it never attempted.

    The span is written onto the returned conventions so that
    :func:`coverage_mask` -- which decides whether a *candidate* may be called new -- uses
    the same bounds, instead of falling back to the span of whatever events survived.
    """
    import dataclasses

    from sage.search.catalogue.record import ExternalCatalogue

    times = catalogue.gps()
    if intervals.size == 0:
        keep = np.zeros(times.size, dtype=bool)
        span = None
    else:
        keep = np.zeros(times.size, dtype=bool)
        for lo, hi in intervals:
            keep |= (times >= lo) & (times < hi)
        span = (float(intervals[:, 0].min()), float(intervals[:, 1].max()))
    return ExternalCatalogue(
        key=catalogue.key,
        events=[e for e, inside in zip(catalogue.events, keep) if inside],
        conventions=dataclasses.replace(
            catalogue.conventions, searched_gps_span=span
        ),
        reference=catalogue.reference,
        version=catalogue.version,
        retrieved_utc=catalogue.retrieved_utc,
    )


def run(spec, **kwargs) -> dict:
    """
    Stage driver: compare the candidate list against everything published.

    This is the step the search exists for. It answers three questions and keeps them
    apart, because they are not the same question:

    - **Did we recover what is already known?** A search that misses published events is
      not ready to claim anything about the ones it does find, so the recovered/missed
      split over the GWOSC list is the validation gate.
    - **Is any candidate absent from every catalogue?** That is what ``is_new`` means,
      and it is only asked where somebody actually searched -- absence outside another
      group's scope is not evidence.
    - **What did each source call it?** Recorded per source under that source's own
      convention. A FAR and a p_astro are different quantities and are never placed on
      one axis; :meth:`Conventions.significance_comparable_to` says which may be read
      against which.

    Sources arrive two ways. GWOSC has a stable API and is fetched through the frozen
    cache, so a campaign re-runs offline against the bytes it used. Everything else --
    another group's catalogue, a sub-threshold list, a glitch list -- comes in as times
    through :func:`sage.search.catalogue.eventlist.read_event_times`, because those
    sources restructure between releases and their layout is not what this uses.

    The candidate table is rewritten with the comparison columns rather than a separate
    file being produced: the table is the single source of truth for everything
    downstream, and a second file holding half the answer is how a figure and a table
    come to disagree.
    """
    from pathlib import Path

    from sage.search.candidates import CandidateTable
    from sage.search.catalogue.cache import CatalogueCache
    from sage.search.catalogue.eventlist import read_event_times
    from sage.search.catalogue.gwosc import load_cumulative, load_marginal
    from sage.search.fingerprint import combine, digest_h5

    table = CandidateTable.load(
        spec.path("candidates", "candidates.h5"), allow_undetermined=True
    )
    tolerance = float(spec.catalogue.match_tolerance_s)

    cache_dir = spec.catalogue.cache_dir or spec.path("catalogue", "cache")
    cache = CatalogueCache(cache_dir, offline_only=bool(spec.catalogue.offline))

    # What this campaign actually analysed. Every catalogue is restricted to it before
    # anything is called recovered or missed: an event outside the analysed time was
    # never looked for, and reporting it as missed makes an O3a search answer for O4.
    analysed = _analysed_intervals(spec)

    catalogues = {}
    gwosc = load_cumulative(cache, endpoint=spec.catalogue.gwtc_endpoint)
    # Scored against the BBH subset. The searched span also holds a BNS and NSBH-class
    # candidates, which are outside what this search looked for -- counting them as
    # misses would penalise the search for not finding what it never searched for.
    catalogues["gwosc"] = _restrict(gwosc.filter_bbh(), analysed)
    if spec.catalogue.include_marginal:
        catalogues["gwosc-marginal"] = _restrict(load_marginal(cache), analysed)

    for entry in spec.catalogue.event_lists or ():
        key, _, path = str(entry).partition("=")
        if not path:
            raise ValueError(
                f"catalogue.event_lists entry {entry!r} is not 'key=path'; the key names "
                "the source in the comparison columns and cannot be guessed from a "
                "filename that may be anything"
            )
        catalogues[key] = _restrict(read_event_times(Path(path), key=key), analysed)

    labels = classify(table, catalogues, tolerance_s=tolerance)
    columns = dict(table.columns)
    for name in ("is_new", "catalogue_match", "catalogue_dt_s", "catalogue_source"):
        columns[name] = labels[name]
    for key in catalogues:
        columns[f"seen_by_{key}"] = labels[f"{key}_matched"]
        columns[f"covered_by_{key}"] = labels[f"{key}_covered"]

    updated = CandidateTable(columns=columns, attrs={**table.attrs, "crossmatched": True})
    target = spec.path("candidates", "candidates.h5")
    updated.save(target)

    comparison = comparison_table(table, catalogues, tolerance_s=tolerance)
    recovery = {
        key: {
            "n_events": int(labels[f"{key}_event_recovered"].size),
            "n_recovered": int(np.count_nonzero(labels[f"{key}_event_recovered"])),
            "missed": [
                str(name)
                for name, found in zip(
                    labels[f"{key}_event_name"], labels[f"{key}_event_recovered"]
                )
                if not found
            ],
        }
        for key in catalogues
    }
    new_mask = np.asarray(labels["is_new"], dtype=bool)
    return {
        "table": str(target),
        "sources": sorted(catalogues),
        "n_candidates": len(table),
        "n_known": int(np.count_nonzero(labels["known"])),
        "n_new": int(np.count_nonzero(new_mask)),
        "new_events": [
            {
                "name": str(columns["name"][i]),
                # The merger time, as a catalogue would quote it -- not the window start.
                "gps": float(merger_times(columns)[i]),
                "ifar_yr": float(columns["ifar_yr"][i]),
                "p_astro": (
                    float(columns["p_astro"][i]) if "p_astro" in columns else None
                ),
            }
            for i in np.flatnonzero(new_mask)
        ],
        "recovery": recovery,
        "n_distinct_events": int(comparison["n_events"]),
        "significance_by_source": comparison["significance_by_source"],
        "fingerprint": combine(len(table), int(new_mask.sum()), digest_h5(target)),
    }
