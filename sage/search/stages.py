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
from pathlib import Path
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
    Stage("injections", ("grid",), "Draw, inject and score the injection set", gpu=True),
    # p_astro takes its signal density from the injections' ranking statistics, not from a
    # sensitive volume: sgwc-1 builds p(x|signal) as a KDE over the injection study's own
    # statistics and p(x|noise) as a KDE over the time-slid background, and <VT> enters
    # neither. Depending on `sensitivity` here would have made p_astro wait on a
    # calculation it does not use, and implied to a reader that it did.
    Stage("pastro", ("far", "injections"), "Rate inference and per-candidate probability"),
    # PyCBC's found/missed sensitive volume (pycbc/sensitivity.py). Optional and off the
    # critical path: nothing downstream consumes it, and no result depends on it.
    Stage(
        "sensitivity",
        ("injections", "far"),
        "Found/missed sensitive volume and distance",
        optional=True,
    ),
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


def descendants(name: str) -> List[Stage]:
    """
    Every stage that depends on ``name``, directly or through another stage.

    The dependency chain downstream of a stage, in dependency order. Re-running a stage
    rebuilds its products, and everything built from those products is then describing
    data that no longer exists -- so this is the set that has to be re-run with it.

    Transitive, not immediate: ``far`` is not in ``release``'s ``depends_on``, but
    ``release`` needs ``figures`` needs ``figdata`` needs ``store`` needs ``catalogue``
    needs ``candidates`` needs ``pastro`` needs ``far``, so rebuilding the false-alarm
    rates invalidates the data release. Following only the direct edge would leave the
    far end of that chain quietly stale.

    Crosses tracks, because the graph does: the follow-up stages depend on core ones.
    """
    stage_by_name(name)
    dependants: Dict[str, List[str]] = {}
    for stage in STAGES:
        for dep in stage.depends_on:
            dependants.setdefault(dep, []).append(stage.name)

    found: set = set()
    frontier = list(dependants.get(name, ()))
    while frontier:
        current = frontier.pop()
        if current in found:
            continue
        found.add(current)
        frontier.extend(dependants.get(current, ()))
    return [stage for stage in resolve_order(sorted(found), include_dependencies=False)]


# Campaign-level record of what has run, relative to the spec's out_dir.
MANIFEST_NAME: str = "manifest.h5"


def manifest_path(spec: SearchSpec):
    """Where a campaign records which stages have completed."""
    return spec.path(MANIFEST_NAME)


def is_complete(spec: SearchSpec, stage: str) -> bool:
    """
    Whether a stage is *recorded* as having completed under the current configuration.

    Answered from the campaign manifest alone, which every stage stamps with the spec hash
    it ran under. Two things have to hold: the stage is recorded, and it was recorded under
    *this* configuration. The second is what makes resumption safe -- reusing a product
    built under a different configuration is silent, and produces results that are wrong
    in a way nothing downstream can see.

    **This does not open the products.** A stage whose output has since been deleted or
    moved still reports complete here; what catches that is the consuming stage failing to
    read it, loudly, at the point of use. The manifest is the record of what ran, not an
    inventory of the directory, and making it the latter would mean this module knowing
    every stage's output layout -- which is the stages' own business.

    A missing manifest means nothing has run, which is the state a fresh campaign starts
    in rather than an error.
    """
    stage_by_name(stage)
    path = manifest_path(spec)
    if not Path(path).is_file():
        return False
    from sage.search.manifest import RunManifest

    recorded = RunManifest(path=Path(path)).summary().get("stages", {})
    entry = recorded.get(stage)
    if not isinstance(entry, dict):
        return False
    if not entry.get("complete", True):
        # A stage that ran a share of its work and said so. The background stage is
        # driven as a SLURM array, and one task reports `collated: False` because the
        # other tasks' shards are not there yet; recording that as complete would let the
        # next stage read a background collated from a fraction of the ladder, with the
        # plan's full livetime in the denominator and every rate in the campaign low.
        return False
    return str(entry.get("spec_hash", "")) == str(spec.hash())


def pending(
    spec: SearchSpec,
    track_name: str = "core",
    skip: Sequence[str] = (),
    include_dependencies: bool = True,
) -> List[Stage]:
    """
    Stages still to run for a track, in dependency order.

    A stage is pending if it has not completed under this configuration, and **also** if
    anything it depends on is pending. The second clause is the one that matters: a
    completed stage whose input is being rebuilt is describing data that will no longer
    exist, and its manifest entry cannot know that. Carrying the pending flag downstream
    is what makes a re-run of an early stage propagate instead of leaving a stale tail.

    Every stage returned can actually be run, in the order returned. That is the contract
    worth having, and it has two consequences:

    ``skip`` drops stages by name, for a campaign that deliberately omits an arm -- an
    optional follow-up, or an injection set that is not being run. Skipping a stage is a
    statement that its products are not wanted, so it does not make its dependants
    pending. But a dependant of a skipped stage cannot run either: :func:`run_stage`
    refuses it, because its input was never built. Those are dropped as well rather than
    returned as work the caller cannot do.

    ``include_dependencies`` governs stages outside the requested track. The follow-up
    track depends on the core one -- ``dataquality`` needs ``candidates``, ``release``
    needs ``figures`` and ``tables`` -- so completing ``followup`` genuinely requires
    running core stages, and by default they are included. Pass ``False`` for the stages
    of this track alone, which answers "what is left of the follow-up" rather than "what
    must I run to finish it".
    """
    unknown = [name for name in skip if name not in _index()]
    if unknown:
        raise ValueError(f"skip names unknown stages {sorted(unknown)}")
    wanted = {stage.name for stage in track(track_name)}
    stale: set = set()
    blocked: set = set()
    out: List[Stage] = []
    for stage in resolve_order(sorted(wanted)):
        if stage.name in skip:
            blocked.add(stage.name)
            continue
        if any(dep in blocked for dep in stage.depends_on):
            # Its input was skipped and never built, so run_stage would refuse it.
            blocked.add(stage.name)
            continue
        upstream_stale = any(dep in stale for dep in stage.depends_on)
        if upstream_stale or not is_complete(spec, stage.name):
            stale.add(stage.name)
            if include_dependencies or stage.name in wanted:
                out.append(stage)
    return out


def recorded_fingerprint(spec: SearchSpec, stage: str) -> Optional[str]:
    """
    The product fingerprint a stage recorded last time it ran, if it recorded one.

    ``None`` both when the stage has never run and when it ran without reporting one --
    the two are the same to a caller deciding whether a re-run changed anything, since
    neither gives it a previous value to compare against.
    """
    stage_by_name(stage)
    path = manifest_path(spec)
    if not Path(path).is_file():
        return None
    from sage.search.manifest import RunManifest

    entry = RunManifest(path=Path(path)).summary().get("stages", {}).get(stage)
    if not isinstance(entry, dict):
        return None
    value = entry.get("fingerprint")
    return None if value is None else str(value)


def _reports_complete(report) -> bool:
    """
    Whether a driver's report claims the whole stage, rather than a share of it.

    Two keys, both meaning "there is more of this stage to run": ``complete`` for a driver
    that says so directly, and ``collated`` for the background stage, whose array tasks
    each score a subset and only the last one finds every shard present. Anything else --
    including a driver that reports neither -- is a whole stage, which is what almost
    every stage is.
    """
    if not isinstance(report, dict):
        return True
    for key in ("complete", "collated"):
        if key in report:
            return bool(report[key])
    return True


#: Where a campaign directory records whose it is.
CLAIM_FILE = "campaign.json"


def claim_campaign_dir(spec: SearchSpec) -> Path:
    """
    Bind ``out_dir`` to one spec hash, and refuse a second.

    A campaign's products live at fixed paths under ``out_dir`` -- ``slides/
    slide_plan.h5``, ``background/bg_inclusive.h5`` -- while its *identity* is the spec
    hash, which the manifest keys its records on. Nothing connected the two, so a spec
    that differed anywhere wrote its products over another campaign's while recording
    them under its own name. The first run of the finished campaign then reported every
    stage incomplete and every product on disk belonged to the second.

    The case that makes this more than hypothetical is the intended workflow.
    ``run_search.py --n-slides 8`` is the documented smoke run and shares its
    configuration -- and therefore its tag, and therefore its ``out_dir`` -- with the
    production campaign it is a smoke test *of*. Running it after the real one silently
    replaced an 82-slide ladder with an 8-slide one.

    So the directory is claimed on first use and checked afterwards. Two campaigns that
    differ in any field a stage reads need two directories, which is a one-line change to
    the tag; the alternative is a shared directory in which the last writer wins and
    nothing on disk says so.

    Returns
    -------
    Path
        The claim file.

    Raises
    ------
    ValueError
        The directory belongs to a different spec.
    """
    import json

    out_dir = Path(spec.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    claim = out_dir / CLAIM_FILE
    current = str(spec.hash())
    if claim.is_file():
        try:
            held = json.loads(claim.read_text())
        except json.JSONDecodeError:
            held = {}
        recorded = str(held.get("spec_hash", ""))
        if recorded and recorded != current:
            raise ValueError(
                f"{out_dir} holds the campaign {recorded[:16]} (tag "
                f"{held.get('tag', '?')!r}, from {held.get('config_module', '?')}) and "
                f"this spec is {current[:16]} (tag {spec.tag!r}). Writing here would "
                "overwrite that campaign's products in place while recording them under "
                "a different name, leaving it reporting every stage incomplete. Give "
                "this one its own tag, or delete the directory if the other campaign is "
                "finished with"
            )
        if recorded == current:
            return claim
    claim.write_text(
        json.dumps(
            {
                "spec_hash": current,
                "tag": str(spec.tag),
                "config_module": str(spec.config_module),
                "observing_run": str(spec.data.observing_run),
                "detectors": list(spec.data.detectors),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return claim


def run_stage(
    spec: SearchSpec, stage: str, cascade: str | bool = "auto", **kwargs
) -> object:
    """
    Dispatch to a stage driver; idempotent and resumable.

    The stage's owning module is imported lazily from :data:`STAGE_MODULES` and its
    ``run(spec, **kwargs)`` called. Lazy because importing this module must not pull torch
    or h5py -- a submit script reads the stage graph to decide what to queue, long before
    anything is scored.

    Dependencies are checked first and the stage is refused if any are incomplete. A stage
    run out of order produces a product from inputs that do not exist yet, and because
    every product is stamped it would look entirely ordinary afterwards.

    The stage's manifest entry is dropped *before* the driver is called and rewritten on
    success. Recording only on success is not enough by itself: a re-run overwrites
    products as it goes, so a crash part way through leaves a half-written product on disk
    beside a manifest entry from the previous run still saying the stage completed under
    this configuration. Dropping first makes that window read as "not run", which is what
    it is. Re-running a stage that succeeds replaces the entry, so the manifest keeps
    describing the products actually on disk.

    The report is recorded under ``report``, with ``stage`` and ``spec_hash`` recorded
    beside it. A driver that returns keys of those names keeps them: they are its own
    findings and overwriting them would silently replace what the stage measured.

    **Re-running a stage invalidates everything downstream of it, unless its product is
    unchanged.** On success the manifest entries of every stage in :func:`descendants` are
    dropped, so the next :func:`pending` returns the whole chain. This is the case the spec
    hash cannot see: fix a bug in ``grid.py``, leave the configuration untouched, re-run
    ``grid``, and the hash is unchanged, so without this the fourteen stages built on the
    old lattice would all still report complete.

    A re-run that produces the same product should not cost the campaign anything, and
    whether it did is a measurement rather than a promise. A driver reports a
    ``fingerprint`` in its report -- any short value that changes if and only if its
    product changed -- and the cascade is skipped when it matches the one recorded last
    time. The stage that wrote the product is the only thing that knows how to summarise it
    cheaply, which is why this is the driver's to compute and not this function's: hashing
    the bytes of a background shard set would cost more than rebuilding it.

    A driver that reports no fingerprint cascades. That is the safe direction and the
    correct default for a stage whose output is not reproducible -- anything seeded from
    the clock, or accumulated across a re-submitted array.

    Parameters
    ----------
    cascade : str or bool
        ``"auto"`` (default) invalidates the chain unless the reported fingerprint matches
        the recorded one. ``True`` always invalidates; ``False`` never does. The booleans
        are overrides for a driver whose fingerprint cannot be trusted in either
        direction, and ``False`` is a promise rather than a measurement: if it is wrong,
        every later product keeps a provenance block saying it was built under this
        configuration, which it was, from an input that has since been replaced, which it
        does not record.

    Returns
    -------
    object
        Whatever the stage driver returned, normally a report dict.
    """
    import importlib

    claim_campaign_dir(spec)
    declared = stage_by_name(stage)
    missing = [
        name
        for name in declared.depends_on
        if not is_complete(spec, name)
    ]
    if missing:
        raise ValueError(
            f"cannot run {stage!r}: it depends on {sorted(missing)}, which "
            f"{'has' if len(missing) == 1 else 'have'} not completed under this "
            "configuration. Run those first, or the product would be built from inputs "
            "that do not exist and would carry a provenance block saying otherwise"
        )
    module_name = STAGE_MODULES[stage]
    module = importlib.import_module(module_name)
    driver = getattr(module, "run", None)
    if driver is None:
        raise NotImplementedError(
            f"stage {stage!r} is owned by {module_name}, which exposes no run(spec, ...) "
            "entry point yet"
        )
    from sage.search.manifest import RunManifest

    # Read before the entry is dropped, so a re-run can be compared with what it replaces.
    previous = recorded_fingerprint(spec, stage)

    path = Path(manifest_path(spec))
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(path=path)
    manifest.drop_stage(stage)

    report = driver(spec, **kwargs)

    payload = {"report": dict(report) if isinstance(report, dict) else repr(report)}
    payload["spec_hash"] = str(spec.hash())
    payload["stage"] = str(stage)
    payload["complete"] = _reports_complete(report)
    fingerprint = (
        report.get("fingerprint") if isinstance(report, dict) else None
    )
    if fingerprint is not None:
        payload["fingerprint"] = str(fingerprint)

    if cascade == "auto":
        # Unchanged product, so nothing downstream is describing data that moved. An
        # absent fingerprint is not a match: it is the absence of a measurement.
        invalidate = fingerprint is None or str(fingerprint) != previous
        if not payload["complete"]:
            # A partial run has no product to compare, and its "fingerprint" describes a
            # share rather than a whole. Cascading is the safe direction.
            invalidate = True
    elif isinstance(cascade, bool):
        invalidate = cascade
    else:
        raise ValueError(
            f"cascade must be 'auto', True or False, got {cascade!r}"
        )

    manifest.record_stage(stage, payload)
    if invalidate:
        for downstream in descendants(stage):
            manifest.drop_stage(downstream.name)
    return report
