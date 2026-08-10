#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : stages.py
Description   : Stage registry and dependency graph.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Two tracks.

The core track runs unattended from a trained network to a candidate list, a sensitivity
measurement and the campaign figures. It is what :func:`sage.search.pipeline.run_search`
executes, and nothing in it depends on per-event work.

The follow-up track characterises individual candidates. It is separate because it is
per-event rather than per-campaign, it needs the parameter-estimation environment, and it
is normally applied to a handful of candidates. It reads the core track's outputs and
writes back into the same campaign store.

Candidate tiers are assigned in the core track from significance and probability alone.
Where a candidate has also been vetted, the follow-up track records that verdict and the
tier is re-derived to account for it. This keeps a full catalogue-grade classification
available without making the core sequence wait on per-event work.
"""

import difflib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from sage.search.spec import SearchSpec

TRACKS: Tuple[str, ...] = ("core", "followup")


@dataclass(frozen=True)
class Stage:
    """One resumable step."""

    name: str
    depends_on: Tuple[str, ...]
    description: str
    track: str = "core"
    gpu: bool = False
    optional: bool = False


CORE_STAGES: Tuple[Stage, ...] = (
    Stage("segments", (), "Coincident segments, vetoes and the livetime decomposition"),
    Stage("grid", ("segments",), "Window lattice and exact analysed time"),
    Stage("zerolag", ("grid",), "Score the observing run", gpu=True),
    Stage("slides", ("grid",), "Stratified lag ladder with measured per-slide livetime"),
    Stage("background", ("slides", "zerolag"), "Score the slides and collate", gpu=True),
    Stage("far", ("background",), "False-alarm rates, tail fit and background validation"),
    Stage("injections", ("grid",), "Score the published injection set", gpu=True),
    Stage("sensitivity", ("injections", "far"), "Found/missed matching and sensitive volume-time"),
    Stage("pastro", ("far", "sensitivity"), "Rate inference and per-candidate probability"),
    Stage(
        "trials",
        ("far",),
        "Per-candidate coverage across the search arms, and the resulting factor",
    ),
    Stage("candidates", ("pastro", "trials"), "Candidate table and provisional tiers"),
    Stage("catalogue", ("candidates",), "Compare against published catalogues"),
    Stage("store", ("catalogue",), "Populate the queryable campaign store"),
    Stage("figdata", ("store",), "Build the figure inputs"),
    Stage("figures", ("figdata",), "Render the campaign figures"),
    Stage("tables", ("store",), "Render the campaign tables"),
)

FOLLOWUP_STAGES: Tuple[Stage, ...] = (
    Stage("dataquality", ("candidates",), "Per-candidate data-quality vetting", track="followup"),
    Stage("qscans", ("candidates",), "Per-candidate spectrograms", track="followup"),
    Stage("followup_mf", ("candidates",), "Matched-filter follow-up", track="followup"),
    Stage("consistency", ("followup_mf",), "Signal-consistency tests", track="followup"),
    Stage("external", ("candidates",), "Independent-pipeline confirmation", track="followup", optional=True),
    Stage("pe", ("candidates",), "Parameter estimation", track="followup", optional=True),
    Stage("skymaps", ("pe",), "Sky localisation", track="followup", optional=True),
    Stage("retier", ("dataquality",), "Re-derive tiers using the vetting verdict", track="followup"),
    Stage("event_pages", ("qscans", "consistency"), "Per-candidate figures", track="followup"),
    Stage("release", ("figures", "tables"), "Data release and machine-readable catalogue", track="followup"),
)

STAGES: Tuple[Stage, ...] = CORE_STAGES + FOLLOWUP_STAGES

# Which module owns each stage. Doubles as the dispatch table for :func:`run_stage`, which
# imports lazily so that importing this module does not pull torch, and as the anchor for
# the test asserting the stage order does not contradict the import graph.
STAGE_MODULES: Dict[str, str] = {
    "segments": "sage.search.segments",
    "grid": "sage.search.grid",
    "zerolag": "sage.search.engine",
    "slides": "sage.search.slides",
    "background": "sage.search.background",
    "far": "sage.search.far",
    "injections": "sage.search.injection.campaign",
    "sensitivity": "sage.search.sensitivity.vt",
    "pastro": "sage.search.pastro.run",
    "trials": "sage.search.trials",
    "candidates": "sage.search.candidates",
    "catalogue": "sage.search.crossmatch",
    "store": "sage.search.store",
    "figdata": "sage.search.figdata",
    "figures": "sage.search.figures",
    "tables": "sage.search.release.tables",
    "dataquality": "sage.search.characterize.dq",
    "qscans": "sage.search.characterize.qscan",
    "followup_mf": "sage.search.characterize.followup_mf",
    "consistency": "sage.search.characterize.consistency",
    "external": "sage.search.characterize.external_pipeline",
    "pe": "sage.search.characterize.pe",
    "skymaps": "sage.search.characterize.skymap",
    "retier": "sage.search.candidates",
    "event_pages": "sage.search.figdata.build_event",
    "release": "sage.search.release.manifest",
}

def _index() -> Dict[str, Stage]:
    """Name-to-stage lookup, built from the registry on each call.

    Deliberately not cached at module scope: a snapshot taken at import time goes stale
    the moment the registry is altered, and it is altered in tests and could be extended
    at runtime. The registry has a few dozen entries, so rebuilding costs nothing worth
    the risk of a lookup that disagrees with the graph it is supposed to describe.
    """
    return {stage.name: stage for stage in STAGES}


def stage_by_name(name: str) -> Stage:
    """Look up a stage, raising with suggestions on a typo."""
    index = _index()
    try:
        return index[name]
    except KeyError:
        close = difflib.get_close_matches(name, index, n=3, cutoff=0.5)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        raise KeyError(f"unknown stage {name!r}{hint}") from None


def track(name: str) -> Tuple[Stage, ...]:
    """Stages belonging to one track."""
    if name not in TRACKS:
        raise ValueError(f"unknown track {name!r}; expected one of {TRACKS}")
    return tuple(stage for stage in STAGES if stage.track == name)


def resolve_order(
    targets: Sequence[str], include_dependencies: bool = True
) -> List[Stage]:
    """
    Order the requested stages, and their dependencies, so each runs after its inputs.

    Parameters
    ----------
    include_dependencies : bool
        When false, only the requested stages are returned, still in dependency order.
        Useful for re-running a chosen few whose inputs are known to be present.

    Raises
    ------
    KeyError
        A target, or something it depends on, is not a known stage.
    ValueError
        The graph contains a cycle. It cannot in the declared registry, but the registry
        is edited by hand and a cycle would otherwise show up as a stage that silently
        never runs.
    """
    requested = [stage_by_name(name) for name in targets]
    if include_dependencies:
        wanted = {stage.name for stage in requested}
        frontier = list(wanted)
        while frontier:
            current = stage_by_name(frontier.pop())
            for dep in current.depends_on:
                if dep not in wanted:
                    wanted.add(dep)
                    frontier.append(dep)
    else:
        wanted = {stage.name for stage in requested}

    ordered: List[Stage] = []
    placed: set = set()
    visiting: set = set()

    def visit(name: str) -> None:
        if name in placed:
            return
        if name in visiting:
            raise ValueError(f"cycle in the stage graph at {name!r}")
        visiting.add(name)
        stage = stage_by_name(name)
        for dep in stage.depends_on:
            if dep in wanted:
                visit(dep)
        visiting.discard(name)
        placed.add(name)
        ordered.append(stage)

    # Registry order breaks ties, so the result is deterministic rather than
    # set-iteration dependent.
    for stage in STAGES:
        if stage.name in wanted:
            visit(stage.name)
    return ordered


def is_complete(spec: SearchSpec, stage: str) -> bool:
    """Whether a stage's products exist and match the current configuration."""
    raise NotImplementedError


def pending(spec: SearchSpec, track_name: str = "core", skip: Sequence[str] = ()) -> List[Stage]:
    """Stages still to run for a track."""
    raise NotImplementedError


def run_stage(spec: SearchSpec, stage: str, **kwargs) -> object:
    """Dispatch to a stage driver; idempotent and resumable."""
    raise NotImplementedError
