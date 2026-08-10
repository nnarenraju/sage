#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : pipeline.py
Description   : Run a complete search from a trained network, end to end.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The single entry point for searching an observing run once a network has been trained.
Given the weights, the configuration they were trained under, and which run to search, it
carries out every step in order and finishes with the campaign figures and tables.

    from sage.search.pipeline import run_search

    result = run_search(
        checkpoint="/work/nagarajan/sage_runs/o4a/CHECKPOINTS/best.pt",
        config_module="runs.o4a.config_HL",
        observing_run="O4a",
    )

What it does, in order: read the network and check its stored geometry against the live
configuration; build the coincident segments and the window lattice; score the run;
cluster; build the time-slide background; assign false-alarm rates and validate the
background; run the injection campaign and measure sensitivity; infer the rates and
assign astrophysical probabilities; assemble the candidate table and compare it against
published catalogues; then build the figure inputs and render the figures.

Per-event work is deliberately outside this sequence. Data-quality vetting, spectrograms,
follow-up filtering, parameter estimation and localisation are run separately against the
candidate list this produces, because they are expensive, need a different software
environment, and are usually applied to a handful of candidates rather than all of them.
See :mod:`sage.search.characterize` and :func:`sage.search.pipeline.run_followup`.

Every step is resumable. Re-running after an interruption skips the steps whose products
are already present and consistent with the configuration, so the same call can be used
to start, resume and extend a campaign.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from sage.search.spec import SearchSpec


@dataclass
class StageOutcome:
    """What happened in one step."""

    stage: str
    status: str
    seconds: float = 0.0
    products: List[Path] = field(default_factory=list)
    summary: Dict[str, object] = field(default_factory=dict)
    message: str = ""


@dataclass
class SearchResult:
    """The outcome of a campaign."""

    spec: SearchSpec
    outcomes: Dict[str, StageOutcome] = field(default_factory=dict)
    candidates: Optional[Path] = None
    store: Optional[Path] = None
    figures: Dict[str, Path] = field(default_factory=dict)
    tables: Dict[str, Path] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether every step completed."""
        raise NotImplementedError

    def n_candidates(self, tier: int = 0) -> int:
        """Candidate count at or above a tier."""
        raise NotImplementedError

    def summary(self) -> str:
        """Readable report: livetimes, counts, sensitivity and where things were written."""
        raise NotImplementedError

    def open_store(self):
        """Open the campaign store for querying."""
        raise NotImplementedError


def run_search(
    checkpoint: Union[str, Path],
    config_module: str,
    observing_run: str,
    out_dir: Optional[Union[str, Path]] = None,
    detectors: Sequence[str] = ("H1", "L1"),
    fiducial_dir: Optional[Union[str, Path]] = None,
    n_slides: int = 82,
    injection_release: Optional[str] = None,
    tag: Optional[str] = None,
    stop_after: Optional[str] = None,
    skip: Sequence[str] = (),
    resume: bool = True,
    dry_run: bool = False,
    **overrides,
) -> SearchResult:
    """
    Search one observing run with a trained network and produce the campaign outputs.

    One call searches one network over one observing run: that pair is an *arm*. Running
    two networks over the same run, such as a two-detector and a three-detector search, is
    two calls, and their candidate lists are combined afterwards with the trials factor
    from :mod:`sage.search.trials`. Nothing here assumes a particular number of detectors;
    quantities that depend on the network geometry, in particular the light-travel time
    that sets the minimum slide lag, are taken as a maximum over every detector pair.

    Parameters
    ----------
    checkpoint : path
        Trained weights. The configuration stored alongside them is checked against the
        live configuration, since a mismatch in window length, sampling rate, detector
        ordering or normalisation would silently invalidate every result.
    config_module : str
        Importable module of the training run, used to reproduce the geometry and the
        preprocessing the network was trained with.
    observing_run : str
        Which run to search. Determines the strain release, the segments, the background
        and every rate the search reports.
    out_dir : path, optional
        Campaign root. Defaults to a directory named for the run and the checkpoint,
        under the shared search area on project storage.
    fiducial_dir : path, optional
        Spectra used for whitening. Defaults to the ``fiducial_dir`` recorded in the
        checkpoint, which is the set the network was trained with. Any other set may be
        given; the choice is the caller's, exactly as it is for a training run. Whichever
        is used is recorded in provenance so a campaign states what it whitened with.
    n_slides : int
        Time slides for the background. This sets how deep the background goes; the
        livetime it actually achieves is measured from the resulting plan.
    injection_release : str, optional
        Published injection set for the sensitivity measurement. Defaults to the one
        covering this observing run.
    stop_after : str, optional
        Stop once this step has completed, for staged running.
    skip : sequence of str
        Steps to leave out. Skipping one whose products are missing will stop anything
        downstream that needs them.
    resume : bool
        Skip steps already completed for this configuration.
    dry_run : bool
        Report the plan, the steps that would run and the expected cost, without running.

    Returns
    -------
    SearchResult
        Per-step outcomes, the candidate table, the campaign store and the figures.

    Notes
    -----
    Per-event characterization and parameter estimation are not part of this sequence;
    run them afterwards with :func:`run_followup`.
    """
    raise NotImplementedError


def build_spec(
    checkpoint: Union[str, Path],
    config_module: str,
    observing_run: str,
    out_dir: Optional[Union[str, Path]] = None,
    detectors: Sequence[str] = ("H1", "L1"),
    fiducial_dir: Optional[Union[str, Path]] = None,
    n_slides: int = 82,
    tag: Optional[str] = None,
    **overrides,
) -> SearchSpec:
    """
    Assemble the campaign specification from a checkpoint and an observing run.

    Fills in the paths and defaults that follow from the run, then validates the whole
    thing before any work begins, so a misconfiguration surfaces immediately rather than
    part-way through a long campaign.
    """
    raise NotImplementedError


def plan(spec: SearchSpec, skip: Sequence[str] = (), resume: bool = True) -> List[str]:
    """Steps that would run, in order, given what is already complete."""
    raise NotImplementedError


def estimate_cost(spec: SearchSpec) -> Dict[str, object]:
    """
    Projected compute, storage and wall time for a campaign.

    Background dominates, and its cost scales with the number of slides, so this is worth
    consulting before committing to a deep run.
    """
    raise NotImplementedError


def run_followup(
    result: Union[SearchResult, SearchSpec],
    candidates: Optional[Sequence[str]] = None,
    tier: int = 1,
    level: str = "screen",
    parameter_estimation: bool = False,
) -> Dict[str, object]:
    """
    Characterise candidates produced by a completed search.

    Separate from :func:`run_search` by design: this is per-event work, it needs the
    parameter-estimation environment, and it is normally applied to a chosen few rather
    than to every candidate.

    Parameters
    ----------
    candidates : sequence of str, optional
        Names to characterise. Defaults to everything at or above ``tier``.
    level : {"screen", "full"}
        Screening covers data quality, spectrograms, the follow-up filter and the
        consistency tests. The full level adds localisation and independent-pipeline
        confirmation.
    parameter_estimation : bool
        Run parameter estimation as well. Much the most expensive step, and submitted
        into its own environment.
    """
    raise NotImplementedError


def main(argv: Optional[list] = None) -> int:
    """Command-line entry point mirroring :func:`run_search`."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
