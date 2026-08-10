#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : trials.py
Description   : Trials factor: how many analyses had a chance at each candidate.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Searching the same data with more than one detector network gives noise more than one
chance to produce a loud event, so a false-alarm rate measured within one network
understates how often the campaign as a whole throws up something that loud. The
correction is a scalar: the rate is multiplied by the number of analyses that had a
chance at that moment.

The count is per candidate, not per campaign. A three-detector network only analyses time
when all three were observing, so a candidate falling where Virgo was down was reachable
by fewer analyses than one in triple-coincident time, and deserves a smaller factor. A
single campaign-wide constant would over-penalise every candidate outside the most
restrictive network's livetime.

Two things are recorded for every candidate and stored separately, because they answer
different questions:

* **covered** -- the analyses whose own analysed segments contain the candidate's time.
  This is what sets the factor: an analysis that could have produced a false alarm there
  counts, whether or not it produced anything.
* **found** -- the analyses that actually produced a trigger. This does not enter the
  factor. It is provenance: it says how many independent looks agree, which is evidence
  about the candidate rather than about the search's false-alarm budget.

The corrected numbers never replace the uncorrected ones. Both are carried side by side
through the candidate table, the store and the release, so a reader can see either view
and the correction is always visible as a factor rather than baked into a number.

**The factor is conservative and deliberately so.** Networks sharing detectors do not have
independent noise: a glitch in Hanford is seen by every analysis that includes Hanford, so
two overlapping networks have fewer than two independent chances. Counting them as two is
an upper bound on the penalty, which errs toward under-claiming significance.

External catalogues are recorded in the same structure but excluded from the factor by
default. Another group re-analysing the same data does not change how often *this* search
produces a false alarm; their coverage is provenance for the newness of a candidate, which
is :mod:`sage.search.crossmatch`'s concern, not this one.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# How the factor is counted. "coverage" is the default and the only one that reflects the
# number of chances noise actually had.
CONVENTIONS: Tuple[str, ...] = ("coverage", "detection", "fixed", "none")

# Columns added to the candidate table. The uncorrected columns keep their names, so
# nothing downstream silently changes meaning when the correction is switched on.
TRIALS_COLUMNS: Tuple[str, ...] = (
    "n_trials",
    "trials_convention",
    "covered_by",
    "found_by",
    "best_arm",
    "far_trials_per_yr",
    "ifar_trials_yr",
    "p_value_trials",
)


@dataclass(frozen=True)
class SearchArm:
    """
    One analysis whose false alarms compete with the others.

    An arm is a detector network searched over one observing run: the HL and HLV searches
    of O3a are two arms. Each has its own background, its own FAR curve and its own
    livetime, which is why the factor cannot be inferred from the candidate table alone.
    """

    key: str
    detectors: Tuple[str, ...]
    observing_run: str
    livetime_s: float = 0.0
    internal: bool = True
    far_curve_path: Optional[Path] = None
    note: str = ""

    def __post_init__(self) -> None:
        """Reject an arm that names no detectors or duplicates one."""
        raise NotImplementedError

    @property
    def n_detectors(self) -> int:
        """Size of the network."""
        raise NotImplementedError


@dataclass
class ArmSegments:
    """The time one arm actually analysed."""

    arm: str
    intervals: np.ndarray
    livetime_s: float = 0.0

    def contains(self, gps: np.ndarray) -> np.ndarray:
        """Whether each time falls inside this arm's analysed segments."""
        raise NotImplementedError

    @classmethod
    def from_grid(cls, arm: str, grid) -> "ArmSegments":
        """
        Take the analysed intervals from a built window lattice.

        Read from the lattice rather than from the observing segments, because the two
        differ: a window needs a whole window's worth of contiguous data, so the analysed
        time is strictly less than the coincident time and is what a false alarm could
        actually have occupied.
        """
        raise NotImplementedError


@dataclass
class TrialsRecord:
    """Which analyses had a chance at one candidate, and which saw it."""

    candidate: str
    gps: float
    covered_by: Tuple[str, ...] = ()
    found_by: Tuple[str, ...] = ()
    best_arm: str = ""
    n_trials: int = 1
    convention: str = "coverage"

    def as_dict(self) -> dict:
        """Flat mapping for the candidate store."""
        raise NotImplementedError

    @property
    def is_multiply_found(self) -> bool:
        """Whether more than one analysis produced a trigger here."""
        raise NotImplementedError


@dataclass
class TrialsModel:
    """The arms of a campaign and the time each of them analysed."""

    arms: Dict[str, SearchArm] = field(default_factory=dict)
    segments: Dict[str, ArmSegments] = field(default_factory=dict)
    convention: str = "coverage"
    fixed_factor: Optional[int] = None

    def add(self, arm: SearchArm, segments: Optional[ArmSegments] = None) -> None:
        """Register an arm, refusing a duplicate key."""
        raise NotImplementedError

    def internal_arms(self) -> Tuple[str, ...]:
        """Arms that count toward the factor."""
        raise NotImplementedError

    def coverage_at(self, gps: np.ndarray) -> List[Tuple[str, ...]]:
        """Which arms analysed each of the given times."""
        raise NotImplementedError

    def describe(self) -> str:
        """Readable statement of the arms and the convention, for the methods section."""
        raise NotImplementedError

    def save(self, path: str | Path) -> Path:
        """Persist the model so a correction can be reproduced or undone."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "TrialsModel":
        """Read a persisted model."""
        raise NotImplementedError


def trials_factor(record: TrialsRecord, model: TrialsModel) -> int:
    """
    The scalar for one candidate.

    Under ``"coverage"`` it is the number of internal arms whose analysed time contains
    the candidate; under ``"detection"`` the number that produced a trigger; under
    ``"fixed"`` a stated constant; under ``"none"`` it is one, which leaves every rate
    unchanged and is how the uncorrected view is produced.

    Returns
    -------
    int
        At least one. A candidate found by an arm is by construction covered by it, so a
        factor of zero indicates coverage was computed from the wrong segments and is
        rejected rather than silently clamped.
    """
    raise NotImplementedError


def build_records(
    candidates,
    model: TrialsModel,
    triggers_by_arm: Optional[Dict[str, object]] = None,
    match_window_s: float = 0.1,
) -> List[TrialsRecord]:
    """
    Record coverage and detection for every candidate.

    Parameters
    ----------
    triggers_by_arm : dict, optional
        Clustered triggers per arm, used to fill ``found_by``. Omitted, coverage is still
        recorded and the factor is still computable, since the factor does not depend on
        who found what.
    match_window_s : float
        How close two arms' triggers must be to count as the same event. Wider than the
        light-travel time across the network, since the arms estimate the time
        independently.
    """
    raise NotImplementedError


def apply(
    candidates,
    records: Sequence[TrialsRecord],
    model: TrialsModel,
    observation_time_s: Optional[float] = None,
):
    """
    Add the corrected columns to a candidate table, leaving the originals intact.

    Writes ``far_trials_per_yr = n_trials * far_per_yr``, the corresponding IFAR, and a
    p-value computed at the corrected rate, alongside ``n_trials`` and the convention that
    produced it. The uncorrected ``far_per_yr`` and ``ifar_yr`` are untouched.

    Notes
    -----
    ``p_astro`` is deliberately not scaled. It is a posterior probability from a rate
    mixture, not a tail probability, so multiplying it by a trials factor is not a defined
    operation. A candidate's ``p_astro`` comes from the model of the arm it was assigned
    to, and the number of arms enters, if at all, through that arm's noise rate rather
    than as a multiplier here.
    """
    raise NotImplementedError


def without_trials(candidates):
    """
    The uncorrected view of a candidate table.

    Returns a table whose significance columns are the single-arm ones, with the corrected
    columns dropped and the convention recorded as ``"none"``, so a comparison of the two
    views is a comparison of tables rather than of column names.
    """
    raise NotImplementedError


def comparison(candidates, records: Sequence[TrialsRecord]) -> dict:
    """
    Both views side by side, per candidate.

    Reports the uncorrected and corrected significance together with the factor applied,
    which is the form the correction should appear in for a reader: a candidate that
    crosses an inclusion threshold in one view and not the other is the thing worth
    seeing, and it is invisible if only one view is published.
    """
    raise NotImplementedError


def assign_best_arm(
    candidates, records: Sequence[TrialsRecord], prefer: str = "ifar"
) -> List[str]:
    """
    Choose the arm each candidate is reported under.

    A candidate found by several arms is quoted once, under the arm that ranked it most
    significantly. That choice is itself a selection over arms, which is precisely what
    the trials factor pays for.
    """
    raise NotImplementedError


def summary(records: Sequence[TrialsRecord], model: TrialsModel) -> dict:
    """
    Campaign-level counts: candidates per factor, per arm, and multiply-found.

    Reported in the methods section so the correction is stated rather than implied by a
    changed number.
    """
    raise NotImplementedError
