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

The count is per candidate, not per campaign. A three-detector network only analyses
time when all three were observing, so a candidate falling where Virgo was down was
reachable by fewer analyses than one in triple-coincident time, and deserves a smaller
factor. A single campaign-wide constant would over-penalise every candidate outside
the most restrictive network's livetime.

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

**The factor is conservative and deliberately so.** Networks sharing detectors do not
have independent noise: a glitch in Hanford is seen by every analysis that includes
Hanford, so two overlapping networks have fewer than two independent chances. Counting
them as two is an upper bound on the penalty, which errs toward under-claiming
significance.

External catalogues are recorded in the same structure but excluded from the factor by
default. Another group re-analysing the same data does not change how often *this*
search produces a false alarm; their coverage is provenance for the newness of a
candidate, which is :mod:`sage.search.crossmatch`'s concern, not this one.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sage.search.fingerprint import combine, digest_values

from sage.search.far import SECONDS_PER_JULIAN_YEAR, p_value_from_ifar
from sage.search.segments import merge_intervals

# How the factor is counted. "coverage" is the default and the only one that reflects
# the number of chances noise actually had.
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

# Arm lists reach the store as one TEXT column per candidate, so they are joined rather
# than stored as a variable-length array. The separator is refused inside an arm key, so
# the join is invertible.
ARM_SEPARATOR: str = ","

# Tier ladder, mirroring :mod:`sage.search.candidates`. Written out rather than
# imported: the candidate table is a later stage than this one in
# :data:`sage.search.stages.STAGES`, so importing it here to obtain three integers would
# invert the stage graph.
TIER_CANDIDATE: int = 0
TIER_CONFIDENT: int = 1
TIER_PE: int = 2

DAYS_PER_JULIAN_YEAR: float = SECONDS_PER_JULIAN_YEAR / 86400.0


@dataclass(frozen=True)
class SearchArm:
    """
    One analysis whose false alarms compete with the others.

    An arm is a detector network searched over one observing run: the HL and HLV
    searches of O3a are two arms. Each has its own background, its own FAR curve and its
    own livetime, which is why the factor cannot be inferred from the candidate table
    alone.
    """

    key: str
    detectors: Tuple[str, ...]
    observing_run: str
    livetime_s: float = 0.0
    internal: bool = True
    far_curve_path: Optional[Path] = None
    note: str = ""

    def __post_init__(self) -> None:
        """
        Reject an arm that names no detectors or duplicates one.

        A repeated detector would be counted twice by anything reading
        :attr:`n_detectors`, and an empty network has no background and no livetime, so
        neither can be a chance that noise had. The key is checked for the separator the
        arm lists are joined with, since a key containing it would split into two arms
        when the candidate table is read back and the factor would not match the record.
        """
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(
            self, "detectors", tuple(str(name) for name in self.detectors)
        )
        if not self.key:
            raise ValueError("an arm must have a key; it names the arm everywhere else")
        if ARM_SEPARATOR in self.key:
            raise ValueError(
                f"arm key {self.key!r} contains {ARM_SEPARATOR!r}, which joins arm "
                "lists in the candidate table; the list could not be read back"
            )
        if not self.detectors:
            raise ValueError(
                f"arm {self.key!r} names no detectors; an empty network has no "
                "background and cannot have had a chance at anything"
            )
        if len(set(self.detectors)) != len(self.detectors):
            raise ValueError(
                f"arm {self.key!r} repeats a detector in {self.detectors}"
            )
        if not np.isfinite(self.livetime_s) or self.livetime_s < 0:
            raise ValueError(
                f"arm {self.key!r} has livetime {self.livetime_s}, which is not a "
                "usable observation time"
            )
        object.__setattr__(self, "livetime_s", float(self.livetime_s))
        object.__setattr__(self, "internal", bool(self.internal))
        if self.far_curve_path is not None:
            object.__setattr__(self, "far_curve_path", Path(self.far_curve_path))

    @property
    def n_detectors(self) -> int:
        """Size of the network."""
        return len(self.detectors)


@dataclass
class ArmSegments:
    """The time one arm actually analysed."""

    arm: str
    intervals: np.ndarray
    livetime_s: float = 0.0

    def __post_init__(self) -> None:
        """
        Normalise the intervals into the sorted, disjoint form membership relies on.

        :meth:`contains` locates a time with one binary search, which is only correct on
        a sorted, non-overlapping list. Sorting here rather than trusting the caller is
        the difference between a wrong answer and an error: an unsorted list silently
        reports "not analysed" for times that were, and those are the candidates whose
        factor would then be too small.

        ``livetime_s`` is filled from the merged intervals only when it was not
        supplied. A caller that knows the exact analysed time -- a lattice, where it is
        an integer window count times the stride -- has a better number than a sum of
        differences of GPS times, which are of order 1.2e9 and carry about 0.24 us of
        resolution each.
        """
        self.arm = str(self.arm)
        if not self.arm:
            raise ValueError("analysed segments must name the arm they belong to")
        intervals = np.asarray(self.intervals, dtype=np.float64)
        if intervals.size == 0:
            intervals = np.empty((0, 2), dtype=np.float64)
        if intervals.ndim != 2 or intervals.shape[1] != 2:
            raise ValueError(
                f"analysed intervals for {self.arm!r} have shape {intervals.shape}; "
                "expected (n, 2) of [start, end)"
            )
        if not np.all(np.isfinite(intervals)):
            raise ValueError(f"analysed intervals for {self.arm!r} are not all finite")
        if np.any(intervals[:, 1] <= intervals[:, 0]):
            raise ValueError(
                f"analysed intervals for {self.arm!r} include one that ends at or "
                "before it starts, which covers no time at all"
            )
        merged = merge_intervals((float(a), float(b)) for a, b in intervals)
        self.intervals = np.asarray(merged, dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(self.livetime_s) or self.livetime_s < 0:
            raise ValueError(
                f"analysed livetime for {self.arm!r} is {self.livetime_s}, which is "
                "not a usable time"
            )
        if self.livetime_s == 0.0:
            self.livetime_s = float(sum(end - start for start, end in merged))

    def contains(self, gps: np.ndarray) -> np.ndarray:
        """
        Whether each time falls inside this arm's analysed segments.

        Intervals are half-open, ``[start, end)``. Closed on both ends a time sitting on
        a boundary would belong to two intervals; here it belongs to the later one,
        which is the same ownership rule the segment sweep uses so that no instant is
        analysed twice.
        """
        times = np.asarray(gps, dtype=np.float64)
        if self.intervals.shape[0] == 0:
            return np.zeros(times.shape, dtype=bool)
        starts = self.intervals[:, 0]
        ends = self.intervals[:, 1]
        # The last interval starting at or before the time; -1 where there is none.
        index = np.searchsorted(starts, times, side="right") - 1
        inside = index >= 0
        safe = np.where(inside, index, 0)
        return inside & (times < ends[safe])

    @classmethod
    def from_grid(cls, arm: str, grid) -> "ArmSegments":
        """
        Take the analysed intervals from a built window lattice.

        Read from the lattice rather than from the observing segments, because the two
        differ: a window needs a whole window's worth of contiguous data, so the
        analysed time is strictly less than the coincident time and is what a false
        alarm could actually have occupied.

        One interval per host span, running from the first window start to one stride
        past the last. Its duration is therefore ``n_windows * stride_s``, which sums to
        the lattice's own livetime; ending at ``last + window_s`` instead would count
        the overlap between neighbouring windows once per window and inflate the
        analysed time by the window-to-stride ratio, a factor of 160 in production.

        The endpoints are computed from the two extreme sample indices of each span
        rather than by materialising every window start: an observing run holds of order
        1e8 windows, and the times alone would be 800 MB for a quantity that is two
        numbers per span.
        """
        stride_s = grid.geometry.stride_s
        rows: List[Tuple[float, float]] = []
        for span in grid.reference_spans:
            if span.n_windows <= 0:
                continue
            last_local = span.first_local + (span.n_windows - 1) * span.stride_samples
            rows.append(
                (span.first_gps, span.segment.gps_of_local(last_local) + stride_s)
            )
        return cls(
            arm=arm,
            intervals=np.asarray(rows, dtype=np.float64).reshape(-1, 2),
            livetime_s=float(grid.livetime_s),
        )


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
    # Ranking of the candidate in each arm that found it, filled by
    # :func:`build_records`. Kept per arm rather than reduced to a single best value,
    # because choosing the arm a candidate is quoted under is itself the selection the
    # trials factor pays for and has to be inspectable after the fact.
    found_stat: Dict[str, float] = field(default_factory=dict)
    found_ifar_yr: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Enforce the invariants the factor depends on.

        The load-bearing one is that every arm that found the candidate also covers it.
        An arm cannot produce a trigger in time it did not analyse, so a violation means
        the coverage was computed from the wrong segments -- and coverage is what sets
        the factor, so the failure would surface as a quietly wrong significance rather
        than as an error.
        """
        self.candidate = str(self.candidate)
        self.gps = float(self.gps)
        self.covered_by = tuple(str(arm) for arm in self.covered_by)
        self.found_by = tuple(str(arm) for arm in self.found_by)
        self.best_arm = str(self.best_arm)
        self.convention = str(self.convention)
        listed = (("covered_by", self.covered_by), ("found_by", self.found_by))
        for label, arms in listed:
            if len(set(arms)) != len(arms):
                raise ValueError(
                    f"{self.candidate!r} lists an arm twice in {label}: {arms}; it "
                    "would be counted twice in the factor"
                )
        missing = [arm for arm in self.found_by if arm not in self.covered_by]
        if missing:
            raise ValueError(
                f"{self.candidate!r} was found by {missing} but those arms are not "
                "recorded as covering it; an arm cannot produce a trigger in time it "
                "did not analyse, so the analysed segments are wrong"
            )
        if self.best_arm and self.best_arm not in self.covered_by:
            raise ValueError(
                f"{self.candidate!r} is quoted under {self.best_arm!r}, which did not "
                "cover it"
            )
        if self.convention not in CONVENTIONS:
            raise ValueError(
                f"unknown trials convention {self.convention!r}; expected one of "
                f"{CONVENTIONS}"
            )
        self.n_trials = int(self.n_trials)
        if self.n_trials < 1:
            raise ValueError(
                f"{self.candidate!r} has a trials factor of {self.n_trials}; a "
                "candidate was reachable by at least the arm that reported it"
            )

    def as_dict(self) -> dict:
        """
        Flat mapping for the candidate store.

        Keys match the ``trials`` table of :mod:`sage.search.store` exactly, so the row
        can be written without a translation step that could drift from the schema. The
        candidate's time is deliberately absent: the store keys this table on the name
        and already holds the time in ``events``, and a second copy is a second thing to
        disagree.

        The rate columns are not here either. They are a product of this record and the
        arm's uncorrected rate, which lives in the candidate table, so they are written
        by :func:`apply` where both are in hand.
        """
        return {
            "name": self.candidate,
            "n_trials": int(self.n_trials),
            "trials_convention": self.convention,
            "covered_by": ARM_SEPARATOR.join(self.covered_by),
            "found_by": ARM_SEPARATOR.join(self.found_by),
            "best_arm": self.best_arm,
        }

    @property
    def is_multiply_found(self) -> bool:
        """Whether more than one analysis produced a trigger here."""
        return len(self.found_by) > 1


@dataclass
class TrialsModel:
    """The arms of a campaign and the time each of them analysed."""

    arms: Dict[str, SearchArm] = field(default_factory=dict)
    segments: Dict[str, ArmSegments] = field(default_factory=dict)
    convention: str = "coverage"
    fixed_factor: Optional[int] = None

    def add(self, arm: SearchArm, segments: Optional[ArmSegments] = None) -> None:
        """
        Register an arm, refusing a duplicate key.

        Refusing rather than replacing, because the second registration of a key is
        almost always a second arm that was meant to be distinct -- HL and HLV of the
        same run, keyed by the run alone -- and silently replacing would drop one arm
        from the factor while leaving a model that looks complete.
        """
        if not isinstance(arm, SearchArm):
            raise TypeError(f"expected a SearchArm, got {type(arm).__name__}")
        if arm.key in self.arms:
            raise ValueError(
                f"arm {arm.key!r} is already registered as "
                f"{self.arms[arm.key].detectors}; two arms sharing a key would count "
                "as one and the factor would be too small"
            )
        if segments is not None and segments.arm != arm.key:
            raise ValueError(
                f"analysed segments for {segments.arm!r} cannot be registered under "
                f"arm {arm.key!r}"
            )
        self.arms[arm.key] = arm
        if segments is not None:
            self.segments[arm.key] = segments

    def internal_arms(self) -> Tuple[str, ...]:
        """
        Arms that count toward the factor.

        In registration order, so ``covered_by`` and every count taken from it are
        reproducible rather than dependent on set iteration.
        """
        return tuple(key for key, arm in self.arms.items() if arm.internal)

    def coverage_at(self, gps: np.ndarray) -> List[Tuple[str, ...]]:
        """
        Which arms analysed each of the given times.

        External arms are included: their coverage bears on whether a candidate is new,
        which is recorded even though :func:`trials_factor` excludes them from the
        count.

        Raises
        ------
        ValueError
            An internal arm has no analysed segments. Treating an unknown analysed time
            as "covered nothing" would make the factor too small for every candidate at
            once, in the direction that overstates significance, and nothing downstream
            would show that an arm had been left out.
        """
        times = np.asarray(gps, dtype=np.float64).ravel()
        blind = [
            key
            for key, arm in self.arms.items()
            if arm.internal and key not in self.segments
        ]
        if blind:
            raise ValueError(
                f"internal arms {blind} have no analysed segments, so their coverage "
                "is unknown; the factor would silently be too small for every candidate"
            )
        masks = {
            key: self.segments[key].contains(times)
            for key in self.arms
            if key in self.segments
        }
        return [
            tuple(key for key, mask in masks.items() if bool(mask[index]))
            for index in range(times.size)
        ]

    def describe(self) -> str:
        """
        Readable statement of the arms and the convention, for the methods section.

        Includes the conservative caveat, because the number this model produces is an
        upper bound on the penalty and a methods section that quotes the factor without
        saying so is claiming an independence the arms do not have.

        Each arm's time is quoted from its analysed segments where it has them and from
        its declared livetime otherwise: coverage is read from the segments, so the time
        they sum to is the one the factor was actually computed against.
        """
        internal = self.internal_arms()
        lines = [
            f"Trials factor: {self.convention} convention over "
            f"{len(internal)} internal arm(s) of {len(self.arms)} registered."
        ]
        for key, arm in self.arms.items():
            segments = self.segments.get(key)
            livetime_s = segments.livetime_s if segments is not None else arm.livetime_s
            role = "internal" if arm.internal else "external, excluded from the factor"
            analysed = (
                f"{livetime_s / 86400.0:.2f} d analysed"
                if livetime_s > 0
                else "analysed time not recorded"
            )
            lines.append(
                f"  {key}: {'+'.join(arm.detectors)} in {arm.observing_run}, "
                f"{analysed}, {role}"
            )
        if self.convention == "fixed":
            lines.append(
                f"  every candidate is given the fixed factor {self.fixed_factor}."
            )
        elif self.convention == "none":
            lines.append("  no correction is applied; rates are single-arm rates.")
        else:
            lines.append(
                "  the factor is per candidate: an arm counts only where its own "
                "analysed segments contain the candidate."
            )
        lines.append(
            "  Arms sharing detectors do not have independent noise, so counting them "
            "separately is an upper bound on the penalty."
        )
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        """
        Persist the model so a correction can be reproduced or undone.

        JSON, not HDF5. The model is small -- a handful of arms and one interval per
        host span, of order a megabyte for an observing run -- and it is the record of a
        decision rather than an array product, so being readable by anything, a reviewer
        with a text editor included, is worth more than the space. Python's float repr is
        shortest-round-trip, so the interval endpoints reload bit-identical and a
        reloaded model reproduces the same coverage.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "convention": self.convention,
            "fixed_factor": self.fixed_factor,
            "arms": [
                {
                    "key": arm.key,
                    "detectors": list(arm.detectors),
                    "observing_run": arm.observing_run,
                    "livetime_s": arm.livetime_s,
                    "internal": arm.internal,
                    "far_curve_path": (
                        None if arm.far_curve_path is None else str(arm.far_curve_path)
                    ),
                    "note": arm.note,
                }
                for arm in self.arms.values()
            ],
            "segments": {
                key: {
                    "livetime_s": segments.livetime_s,
                    "intervals": segments.intervals.tolist(),
                }
                for key, segments in self.segments.items()
            },
        }
        body = json.dumps(payload, indent=2, sort_keys=False)
        path.write_text(body, encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrialsModel":
        """
        Read a persisted model.

        Rebuilt through :meth:`add`, so a stored model is subject to the same checks a
        live one is: a file edited into an inconsistent state is refused here rather
        than producing a factor that no arm list supports.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            convention=str(payload.get("convention", "coverage")),
            fixed_factor=payload.get("fixed_factor"),
        )
        stored = payload.get("segments", {})
        for entry in payload.get("arms", []):
            arm = SearchArm(
                key=entry["key"],
                detectors=tuple(entry["detectors"]),
                observing_run=entry["observing_run"],
                livetime_s=float(entry.get("livetime_s", 0.0)),
                internal=bool(entry.get("internal", True)),
                far_curve_path=entry.get("far_curve_path"),
                note=str(entry.get("note", "")),
            )
            segments = stored.get(arm.key)
            model.add(
                arm,
                None
                if segments is None
                else ArmSegments(
                    arm=arm.key,
                    intervals=np.asarray(
                        segments["intervals"], dtype=np.float64
                    ).reshape(-1, 2),
                    livetime_s=float(segments.get("livetime_s", 0.0)),
                ),
            )
        if model.convention not in CONVENTIONS:
            raise ValueError(
                f"stored model names convention {model.convention!r}, which is not one "
                f"of {CONVENTIONS}"
            )
        return model


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
    if model.convention not in CONVENTIONS:
        raise ValueError(
            f"unknown trials convention {model.convention!r}; expected one of "
            f"{CONVENTIONS}"
        )
    unknown = [arm for arm in record.covered_by if arm not in model.arms]
    if unknown:
        raise ValueError(
            f"{record.candidate!r} is covered by arms {unknown} that this model does "
            "not register; the record and the model describe different campaigns"
        )
    if model.convention == "none":
        return 1
    if model.convention == "fixed":
        if _unusable_fixed_factor(model.fixed_factor):
            raise ValueError(
                f"the fixed convention needs a stated factor of at least one, got "
                f"{model.fixed_factor!r}"
            )
        return int(model.fixed_factor)

    internal = set(model.internal_arms())
    if model.convention == "detection":
        counted = [arm for arm in record.found_by if arm in internal]
        if not counted:
            raise ValueError(
                f"{record.candidate!r} was found by no internal arm, so the detection "
                "convention has nothing to count; a candidate is reported by the arm "
                "that produced it"
            )
        return len(counted)
    counted = [arm for arm in record.covered_by if arm in internal]
    if not counted:
        raise ValueError(
            f"{record.candidate!r} at {record.gps} is covered by no internal arm; "
            "every candidate came from an arm that analysed that time, so the analysed "
            "segments are wrong"
        )
    return len(counted)


def _unusable_fixed_factor(factor) -> bool:
    """
    Whether a stated fixed factor cannot be a count of chances.

    A bool is rejected explicitly: ``True`` is an integer of value one in Python, so a
    model configured with ``fixed_factor=True`` would otherwise apply a factor of one
    and look as though it had been corrected.
    """
    return (
        factor is None
        or isinstance(factor, bool)
        or not isinstance(factor, (int, np.integer))
        or int(factor) < 1
    )


def build_records(
    candidates,
    model: TrialsModel,
    triggers_by_arm: Optional[Dict[str, object]] = None,
    match_window_s: float = 0.1,
    curves: Optional[Dict[str, object]] = None,
) -> List[TrialsRecord]:
    """
    Record coverage and detection for every candidate.

    Parameters
    ----------
    candidates : CandidateTable or mapping
        Anything carrying columnar ``name`` and ``gps``. The window-start time is used,
        not a coalescence-time estimate: coverage asks which arms analysed a window
        there, and the lattice is expressed in window starts.
    triggers_by_arm : dict, optional
        Clustered triggers per arm, used to fill ``found_by``. Omitted, coverage is
        still recorded and the factor is still computable, since the factor does not
        depend on who found what.
    match_window_s : float
        How close two arms' triggers must be to count as the same event. Wider than the
        light-travel time across the network, since the arms estimate the time
        independently.
    curves : dict, optional
        Per-arm :class:`~sage.search.far.FarCurve`, used to record what each finding arm
        ranked the candidate at. Passed in memory rather than loaded from
        :attr:`SearchArm.far_curve_path`, because the curve already built in this
        process is the one that produced the candidate's own rate; re-reading the path
        could rank the arms with a curve different from the one behind ``far_per_yr``.

    Returns
    -------
    list of TrialsRecord
        One per candidate row, in table order, each carrying the factor its convention
        gives and the arm it would be quoted under.
    """
    columns = _columns(candidates)
    names = _names(columns)
    times = _times(columns)
    if not np.isfinite(match_window_s) or match_window_s <= 0:
        raise ValueError(
            f"match_window_s must be finite and positive, got {match_window_s}"
        )
    coverage = model.coverage_at(times)
    triggers = {
        str(arm): _trigger_times_stats(arm, value)
        for arm, value in (triggers_by_arm or {}).items()
    }
    unknown = [arm for arm in triggers if arm not in model.arms]
    if unknown:
        raise ValueError(
            f"triggers were given for arms {unknown} that the model does not register"
        )
    curves = {str(arm): curve for arm, curve in (curves or {}).items()}

    records: List[TrialsRecord] = []
    for index, name in enumerate(names):
        covered = coverage[index]
        if not covered:
            raise ValueError(
                f"candidate {name!r} at {times[index]} lies in no arm's analysed time; "
                "it was produced by an arm that analysed that moment, so the segments "
                "handed to this model are not the ones the search ran on"
            )
        found_stat: Dict[str, float] = {}
        found_ifar: Dict[str, float] = {}
        for arm in model.arms:
            if arm not in triggers:
                continue
            arm_times, arm_stats = triggers[arm]
            close = np.abs(arm_times - times[index]) <= match_window_s
            if not np.any(close):
                continue
            if arm not in covered:
                raise ValueError(
                    f"arm {arm!r} produced a trigger within {match_window_s} s of "
                    f"{name!r} at {times[index]}, but its analysed segments do not "
                    "contain that time; an arm cannot trigger on time it did not "
                    "analyse"
                )
            # The loudest matching trigger, not the nearest: the arm ranks by statistic
            # and the candidate is quoted by rank, so the nearest could quote a quieter
            # trigger than the one the arm actually reported.
            best = int(np.argmax(np.where(close, arm_stats, -np.inf)))
            found_stat[arm] = float(arm_stats[best])
            curve = curves.get(arm)
            if curve is not None:
                found_ifar[arm] = float(
                    np.asarray(curve.ifar_of(np.asarray([arm_stats[best]]))).ravel()[0]
                )
        record = TrialsRecord(
            candidate=name,
            gps=float(times[index]),
            covered_by=covered,
            found_by=tuple(arm for arm in model.arms if arm in found_stat),
            convention=model.convention,
            found_stat=found_stat,
            found_ifar_yr=found_ifar,
        )
        record.n_trials = trials_factor(record, model)
        record.best_arm = _best_arm(record)
        records.append(record)
    return records


def apply(
    candidates,
    records: Sequence[TrialsRecord],
    model: TrialsModel,
    observation_time_s: Optional[float] = None,
    p_astro_confident: float = 0.5,
    far_pe_per_yr: float = 1.0,
):
    """
    Add the corrected columns to a candidate table, leaving the originals intact.

    Writes ``far_trials_per_yr = n_trials * far_per_yr``, the corresponding IFAR, and a
    p-value computed at the corrected rate, alongside ``n_trials`` and the convention
    that produced it. The uncorrected ``far_per_yr`` and ``ifar_yr`` are untouched.

    ``ifar_trials_yr`` is ``ifar_yr / n_trials`` rather than ``1 / far_trials_per_yr``.
    The two agree to a few ulp but not exactly, and the stored IFAR is also the capped
    one -- :meth:`~sage.search.far.FarCurve.ifar_of` limits it to the length of the
    background that measured it -- so re-deriving it from the rate would quietly undo
    the cap for the corrected view alone.

    ``tier_trials`` is derived from the corrected rate by the same rule that gives
    ``tier`` from the uncorrected one, and ``tier`` is filled the same way when the
    table does not already carry it. Neither is demoted below :data:`TIER_CANDIDATE`:
    dropping a candidate from the list is an inclusion decision belonging to whoever
    assembles the list, it is reported by :func:`comparison`, and writing it as an
    undetermined tier would be read by the store as "no tier assigned yet".

    Parameters
    ----------
    observation_time_s : float, optional
        Time the p-values are quoted over. Omitted, each candidate uses the foreground
        livetime of the arm it is quoted under, which is the time the analysis that
        reported it actually ran; a single campaign-wide time would be the union of arms
        that never analysed the same seconds.

    Notes
    -----
    ``p_astro`` is deliberately not scaled. It is a posterior probability from a rate
    mixture, not a tail probability, so multiplying it by a trials factor is not a
    defined operation. A candidate's ``p_astro`` comes from the model of the arm it was
    assigned to, and the number of arms enters, if at all, through that arm's noise rate
    rather than as a multiplier here.
    """
    columns = _columns(candidates)
    names = _names(columns)
    by_name = _records_by_name(records)
    if "far_per_yr" not in columns:
        raise KeyError(
            "the candidate table carries no far_per_yr; the correction is a factor on "
            "a rate and has nothing to multiply"
        )

    far = np.asarray(columns["far_per_yr"], dtype=np.float64)
    if "ifar_yr" in columns:
        ifar = np.asarray(columns["ifar_yr"], dtype=np.float64)
    else:
        # Only when the table never carried one; a stored IFAR is preferred because it
        # may be capped and this reconstruction is not.
        with np.errstate(divide="ignore"):
            ifar = np.where(far > 0, 1.0 / far, np.inf)

    count = len(names)
    n_trials = np.empty(count, dtype=np.int64)
    conventions: List[str] = []
    covered: List[str] = []
    found: List[str] = []
    best: List[str] = []
    observation = np.empty(count, dtype=np.float64)
    for index, name in enumerate(names):
        record = by_name.get(name)
        if record is None:
            raise KeyError(
                f"no trials record for candidate {name!r}; every candidate needs a "
                "factor, and a missing one would leave the row silently uncorrected"
            )
        n_trials[index] = trials_factor(record, model)
        conventions.append(model.convention)
        covered.append(ARM_SEPARATOR.join(record.covered_by))
        found.append(ARM_SEPARATOR.join(record.found_by))
        best.append(record.best_arm)
        observation[index] = _observation_time(record, model, observation_time_s)

    updated = dict(columns)
    updated["n_trials"] = n_trials
    updated["trials_convention"] = np.asarray(conventions)
    updated["covered_by"] = np.asarray(covered)
    updated["found_by"] = np.asarray(found)
    updated["best_arm"] = np.asarray(best)
    updated["far_trials_per_yr"] = n_trials * far
    updated["ifar_trials_yr"] = ifar / n_trials
    updated["p_value_trials"] = _p_values(updated["ifar_trials_yr"], observation)
    if "p_value" not in columns:
        updated["p_value"] = _p_values(ifar, observation)
    if "ifar_yr" not in columns:
        updated["ifar_yr"] = ifar

    p_astro = columns.get("p_astro")
    if "tier" not in columns:
        updated["tier"] = assign_tiers(far, p_astro, p_astro_confident, far_pe_per_yr)
    updated["tier_trials"] = assign_tiers(
        updated["far_trials_per_yr"], p_astro, p_astro_confident, far_pe_per_yr
    )
    return _rebuilt(candidates, updated)


def without_trials(candidates):
    """
    The uncorrected view of a candidate table.

    Returns a table whose significance columns are the single-arm ones, with the
    corrected columns dropped and the convention recorded as ``"none"``, so a comparison
    of the two views is a comparison of tables rather than of column names.

    Coverage stays. ``covered_by`` and ``found_by`` are properties of the data and of
    the campaign, not of the correction, and a reader of the uncorrected table still
    needs to see how many arms were looking. What is removed is every number the factor
    changed, and ``n_trials`` is set to one rather than dropped, so the table states
    which view it is instead of leaving it to be inferred from an absent column.
    """
    columns = _columns(candidates)
    count = len(_names(columns))
    stripped = {
        name: values
        for name, values in columns.items()
        if name
        not in (
            "far_trials_per_yr",
            "ifar_trials_yr",
            "p_value_trials",
            "tier_trials",
            "n_trials",
            "trials_convention",
        )
    }
    stripped["n_trials"] = np.ones(count, dtype=np.int64)
    stripped["trials_convention"] = np.asarray(["none"] * count)
    return _rebuilt(candidates, stripped)


def comparison(
    candidates,
    records: Sequence[TrialsRecord],
    far_include_per_day: float = 2.0,
) -> dict:
    """
    Both views side by side, per candidate.

    Reports the uncorrected and corrected significance together with the factor applied,
    which is the form the correction should appear in for a reader: a candidate that
    crosses an inclusion threshold in one view and not the other is the thing worth
    seeing, and it is invisible if only one view is published.

    Parameters
    ----------
    far_include_per_day : float
        Inclusion threshold for the public candidate list, in false alarms per day. The
        default matches the broad tier of :func:`sage.search.candidates.apply_tiers`.
        The comparison is strict, so a candidate sitting exactly on the threshold is
        excluded in both views and is not reported as a crossing.

    Returns
    -------
    dict
        Both views as arrays in table order, plus ``crossings``, the names admitted by
        one view and not the other, and ``tier_changed`` where the corrected tier
        differs from the uncorrected one.
    """
    columns = _columns(candidates)
    names = _names(columns)
    by_name = _records_by_name(records)
    missing = [name for name in names if name not in by_name]
    if missing:
        raise KeyError(f"no trials record for candidates {missing[:5]}")
    for required in ("far_per_yr", "far_trials_per_yr"):
        if required not in columns:
            raise KeyError(
                f"the candidate table carries no {required}; both views are needed to "
                "compare them, so run trials.apply first"
            )

    far = np.asarray(columns["far_per_yr"], dtype=np.float64)
    far_trials = np.asarray(columns["far_trials_per_yr"], dtype=np.float64)
    threshold = float(far_include_per_day) * DAYS_PER_JULIAN_YEAR
    included = far < threshold
    included_trials = far_trials < threshold
    crossed = included != included_trials

    tier = columns.get("tier")
    tier_trials = columns.get("tier_trials")
    if tier is not None and tier_trials is not None:
        changed = np.asarray(tier) != np.asarray(tier_trials)
    else:
        changed = np.zeros(len(names), dtype=bool)

    out = {
        "name": np.asarray(names),
        "n_trials": np.asarray(
            [by_name[name].n_trials for name in names], dtype=np.int64
        ),
        "covered_by": np.asarray(
            [ARM_SEPARATOR.join(by_name[name].covered_by) for name in names]
        ),
        "far_per_yr": far,
        "far_trials_per_yr": far_trials,
        "included": included,
        "included_trials": included_trials,
        "crossings": tuple(np.asarray(names)[crossed].tolist()),
        "n_crossings": int(crossed.sum()),
        "tier_changed": tuple(np.asarray(names)[changed].tolist()),
        "far_include_per_yr": threshold,
    }
    for name in ("ifar_yr", "ifar_trials_yr", "p_value", "p_value_trials", "tier",
                 "tier_trials"):
        if name in columns:
            out[name] = np.asarray(columns[name])
    return out


def assign_best_arm(
    candidates, records: Sequence[TrialsRecord], prefer: str = "ifar"
) -> List[str]:
    """
    Choose the arm each candidate is reported under.

    A candidate found by several arms is quoted once, under the arm that ranked it most
    significantly. That choice is itself a selection over arms, which is precisely what
    the trials factor pays for.

    Parameters
    ----------
    prefer : {"ifar", "stat"}
        What "most significantly" means. IFAR is the default and the right answer: the
        arms have different backgrounds, so the same ranking statistic is not the same
        significance in two of them, and comparing raw statistics would systematically
        favour whichever arm has the noisier background. ``"stat"`` is available for a
        model whose FAR curves are not yet built, and falls back to automatically when
        no arm recorded an IFAR.

    Returns
    -------
    list of str
        One arm key per candidate row, empty where no arm produced a trigger.
    """
    if prefer not in ("ifar", "stat"):
        raise ValueError(f"prefer must be 'ifar' or 'stat', got {prefer!r}")
    columns = _columns(candidates)
    by_name = _records_by_name(records)
    out: List[str] = []
    for name in _names(columns):
        record = by_name.get(name)
        if record is None:
            raise KeyError(f"no trials record for candidate {name!r}")
        out.append(_best_arm(record, prefer=prefer))
    return out


def summary(records: Sequence[TrialsRecord], model: TrialsModel) -> dict:
    """
    Campaign-level counts: candidates per factor, per arm, and multiply-found.

    Reported in the methods section so the correction is stated rather than implied by a
    changed number.
    """
    records = list(records)
    by_factor: Dict[int, int] = {}
    covered_by_arm = {key: 0 for key in model.arms}
    found_by_arm = {key: 0 for key in model.arms}
    multiply_found: List[str] = []
    for record in records:
        factor = int(record.n_trials)
        by_factor[factor] = by_factor.get(factor, 0) + 1
        for arm in record.covered_by:
            covered_by_arm[arm] = covered_by_arm.get(arm, 0) + 1
        for arm in record.found_by:
            found_by_arm[arm] = found_by_arm.get(arm, 0) + 1
        if record.is_multiply_found:
            multiply_found.append(record.candidate)
    return {
        "n_candidates": len(records),
        "convention": model.convention,
        "arms": tuple(model.arms),
        "internal_arms": model.internal_arms(),
        "by_factor": {factor: by_factor[factor] for factor in sorted(by_factor)},
        "covered_by_arm": covered_by_arm,
        "found_by_arm": found_by_arm,
        "n_multiply_found": len(multiply_found),
        "multiply_found": tuple(multiply_found),
        "describe": model.describe(),
    }


# --------------------------------------------------------------------- table plumbing
def _columns(table) -> Dict[str, np.ndarray]:
    """
    Columns of a candidate table, whatever shape the caller holds it in.

    Accepts a :class:`~sage.search.candidates.CandidateTable` or a plain columnar
    mapping. Duck-typed rather than importing the table class, so this module stays a
    stage earlier than the candidate table in :data:`sage.search.stages.STAGES` and can
    be tested without one.
    """
    columns = getattr(table, "columns", table)
    if not isinstance(columns, dict):
        raise TypeError(
            "expected a candidate table or a mapping of columns, got "
            f"{type(table).__name__}"
        )
    return {str(name): np.asarray(values) for name, values in columns.items()}


def _rebuilt(table, columns: Dict[str, np.ndarray]):
    """
    A table of the caller's own type carrying the new columns.

    The input is never mutated: the correction has to be reversible and inspectable, and
    a caller holding the uncorrected table after calling :func:`apply` must still be
    holding the uncorrected table.
    """
    if getattr(table, "columns", None) is None:
        return columns
    return type(table)(columns=columns, attrs=dict(getattr(table, "attrs", {}) or {}))


def _names(columns: Dict[str, np.ndarray]) -> List[str]:
    """Candidate names as text, decoding the bytes an HDF5 read returns."""
    if "name" not in columns:
        raise KeyError(
            "the candidate table carries no name column; the trials records are keyed "
            "on it and cannot be matched positionally, since the two are built by "
            "different stages"
        )
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in columns["name"].tolist()
    ]


def _times(columns: Dict[str, np.ndarray]) -> np.ndarray:
    """Candidate times, refusing a table that carries none."""
    if "gps" not in columns:
        raise KeyError(
            "the candidate table carries no gps column; coverage is a statement about "
            "a moment in time and cannot be established without one"
        )
    times = np.asarray(columns["gps"], dtype=np.float64).ravel()
    if not np.all(np.isfinite(times)):
        raise ValueError("candidate times must all be finite")
    return times


def _records_by_name(records: Sequence[TrialsRecord]) -> Dict[str, TrialsRecord]:
    """Index records by candidate, refusing two records for one candidate."""
    out: Dict[str, TrialsRecord] = {}
    for record in records:
        if record.candidate in out:
            raise ValueError(
                f"two trials records name candidate {record.candidate!r}; the factor "
                "would depend on which one was read"
            )
        out[record.candidate] = record
    return out


def _trigger_times_stats(arm: str, value) -> Tuple[np.ndarray, np.ndarray]:
    """
    Times and ranking statistics of one arm's clustered triggers.

    Accepts a :class:`~sage.search.triggers.TriggerTable`, a
    :class:`~sage.search.cluster.ClusterResult` or a columnar mapping, since the arm's
    triggers reach this stage from whichever of those the caller kept.
    """
    columns = getattr(value, "columns", None)
    if isinstance(columns, dict) and ("gps" in columns or "tc_gps" in columns):
        times = columns["gps"] if "gps" in columns else columns["tc_gps"]
        stats = columns["stat"]
    elif isinstance(value, dict) and ("gps" in value or "tc_gps" in value):
        times = value["gps"] if "gps" in value else value["tc_gps"]
        stats = value["stat"]
    elif hasattr(value, "times") and hasattr(value, "stats"):
        times, stats = value.times, value.stats
    else:
        raise TypeError(
            f"triggers for arm {arm!r} carry neither gps/stat columns nor times/stats "
            f"arrays; got {type(value).__name__}"
        )
    times = np.asarray(times, dtype=np.float64).ravel()
    stats = np.asarray(stats, dtype=np.float64).ravel()
    if times.size != stats.size:
        raise ValueError(
            f"arm {arm!r} has {times.size} trigger times against {stats.size} "
            "statistics"
        )
    if np.isnan(stats).any():
        raise ValueError(
            f"arm {arm!r} has NaN ranking statistics, which lose every comparison and "
            "would silently never be chosen as the best arm"
        )
    return times, stats


def _best_arm(record: TrialsRecord, prefer: str = "ifar") -> str:
    """
    Arm a record would be quoted under, or ``""`` when no arm found it.

    Ties go to the earlier arm in ``found_by``, which :func:`build_records` fills in the
    model's registration order, so the choice is reproducible rather than dependent on
    dictionary iteration.
    """
    if not record.found_by:
        return ""
    ranking = record.found_ifar_yr if prefer == "ifar" else record.found_stat
    if not ranking:
        ranking = record.found_stat if prefer == "ifar" else record.found_ifar_yr
    if not ranking:
        if len(record.found_by) == 1:
            return record.found_by[0]
        raise ValueError(
            f"{record.candidate!r} was found by {record.found_by} but no arm recorded "
            "a ranking, so the arm it is quoted under cannot be chosen; pass the arms' "
            "triggers to build_records"
        )
    ordered = [arm for arm in record.found_by if arm in ranking]
    if not ordered:
        raise ValueError(
            f"{record.candidate!r} has rankings for arms that did not find it"
        )
    return max(ordered, key=lambda arm: (ranking[arm], -ordered.index(arm)))


def _observation_time(
    record: TrialsRecord, model: TrialsModel, observation_time_s: Optional[float]
) -> float:
    """
    Time a candidate's p-value is quoted over.

    The arm's own foreground livetime when the caller states none, because the p-value
    asks how often the analysis that reported the candidate produces something this loud
    in the time it ran. Refused rather than defaulted when neither is available: a
    p-value quoted over a made-up observation time is a number nothing supports.
    """
    if observation_time_s is not None:
        if not np.isfinite(observation_time_s) or observation_time_s < 0:
            raise ValueError(
                f"observation time must be finite and non-negative, got "
                f"{observation_time_s}"
            )
        return float(observation_time_s)
    if not record.best_arm:
        raise ValueError(
            f"{record.candidate!r} names no arm to take an observation time from; pass "
            "observation_time_s, or give build_records the arms' triggers so the arm "
            "each candidate is quoted under is recorded"
        )
    arm = model.arms.get(record.best_arm)
    if arm is not None and arm.livetime_s > 0:
        return float(arm.livetime_s)
    segments = model.segments.get(record.best_arm)
    if segments is not None and segments.livetime_s > 0:
        return float(segments.livetime_s)
    raise ValueError(
        f"arm {record.best_arm!r} records no livetime, so a p-value over it cannot be "
        "quoted; pass observation_time_s"
    )


def _p_values(ifar_yr: np.ndarray, observation_s: np.ndarray) -> np.ndarray:
    """
    Single-trial p-value per candidate, each over its own observation time.

    :func:`sage.search.far.p_value_from_ifar` takes one observation time, so the per-arm
    times are applied one group at a time rather than by re-deriving the expression here
    -- the underflow-safe form lives there and should have exactly one implementation.
    """
    ifar_yr = np.asarray(ifar_yr, dtype=np.float64)
    observation_s = np.broadcast_to(
        np.asarray(observation_s, dtype=np.float64), ifar_yr.shape
    )
    out = np.empty(ifar_yr.shape, dtype=np.float64)
    for value in np.unique(observation_s):
        rows = observation_s == value
        out[rows] = p_value_from_ifar(ifar_yr[rows], float(value))
    return out


def assign_tiers(
    far_per_yr: np.ndarray,
    p_astro: Optional[np.ndarray],
    p_astro_confident: float,
    far_pe_per_yr: float,
) -> np.ndarray:
    """
    Tier ladder from a rate and an astrophysical probability.

    **The single owner of the tier rule.** :func:`sage.search.candidates.apply_tiers`
    calls this rather than restating it, so the uncorrected ``tier`` and the corrected
    ``tier_trials`` differ only through the factor and not through the rule. Two
    implementations of a threshold ladder is how a candidate ends up in different tiers
    in the table and in the trials comparison, with nothing to say which is right.

    Applied here to whichever rate is being tiered. Thresholds are strict, so a candidate
    exactly on a boundary is excluded.

    Without a ``p_astro`` column no row is promoted past :data:`TIER_CANDIDATE`: the
    confident tier is defined by a probability the table does not carry, and treating an
    absent probability as passing would promote every loud candidate on rate alone.
    """
    far = np.asarray(far_per_yr, dtype=np.float64)
    if p_astro is None:
        probability = np.full(far.shape, np.nan, dtype=np.float64)
    else:
        probability = np.asarray(p_astro, dtype=np.float64)
    # NaN loses every comparison, which is the intended behaviour for a missing p_astro.
    confident = probability > float(p_astro_confident)
    tiers = np.full(far.shape, TIER_CANDIDATE, dtype=np.int64)
    tiers = np.where(confident, TIER_CONFIDENT, tiers)
    return np.where(confident & (far < float(far_pe_per_yr)), TIER_PE, tiers)


def analysed_intervals(spec) -> Tuple[np.ndarray, float]:
    """
    One arm's analysed lattice, as ``(intervals, livetime_s)``.

    Read from the window lattice, not from the observing segments. A window needs a whole
    window of contiguous data in every detector of the network, so analysed time is
    strictly less than coincident time -- on the real O3a release by 11,632 windows. The
    larger number would credit an arm with a chance at a moment it could not have
    triggered on, and every such moment lowers a candidate's factor.

    The livetime is returned as the window count times the stride rather than as a sum of
    interval widths. It is an exact integer multiple, where the sum is of differences of
    GPS times of order 1.2e9 that carry about 0.24 microseconds of resolution each.
    """
    from sage.search.grid import AnalysisGrid
    from sage.search.segments import coincident_intervals, load_segments

    geometry = spec.geometry_object()
    release = Path(spec.data.release_dir)
    segments = {
        detector: load_segments(
            release / f"data_{detector}_{spec.data.observing_run}_segments.json"
        )
        for detector in spec.data.detectors
    }
    grid = AnalysisGrid.build(
        geometry,
        segments,
        coincident_intervals(segments),
        reference_detector=spec.slides.reference_detector,
        coverage=False,
    )
    stride = float(geometry.stride_s)
    spans = grid.reference_spans
    intervals = np.array(
        [
            (float(span.starts_gps()[0]), float(span.starts_gps()[-1]) + stride)
            for span in spans
            if span.n_windows
        ],
        dtype=np.float64,
    ).reshape(-1, 2)
    return intervals, float(len(grid) * stride)


def build_model(spec, sibling_specs: Optional[Sequence[object]] = None) -> TrialsModel:
    """
    Assemble the campaign's arms: this one, plus every sibling that competes with it.

    A sibling is another campaign over the same observing run -- the HLV search beside the
    HL one. Each is a separate spec with its own release, its own background and its own
    lattice, which is why the factor cannot be read off a candidate table.

    A sibling's FAR curve is required to be present. The curve is what the candidate would
    have been ranked at had that arm found it, and an arm registered without one is an arm
    whose competing analysis has not actually been run: counting it as a chance noise had
    would inflate every factor on the strength of an analysis that does not exist.
    """
    from sage.search.far import FarCurve  # noqa: F401  (presence check only)

    model = TrialsModel(
        convention=str(spec.trials.convention),
        fixed_factor=spec.trials.fixed_factor,
    )
    for index, arm_spec in enumerate((spec, *(sibling_specs or ()))):
        if arm_spec.data.observing_run != spec.data.observing_run:
            raise ValueError(
                f"sibling arm {arm_spec.arm!r} searches "
                f"{arm_spec.data.observing_run} while this campaign searches "
                f"{spec.data.observing_run}; arms compete only over the same data, and "
                "a factor built across runs would penalise a candidate for chances that "
                "were taken on different seconds"
            )
        curve = arm_spec.path(
            "far", f"far_curve_{arm_spec.data.observing_run}_inclusive.h5"
        )
        if not Path(curve).is_file():
            raise FileNotFoundError(
                f"arm {arm_spec.arm!r} has no inclusive FAR curve at {curve}; it has "
                "not completed its own `far` stage, so it has neither an analysed "
                "lattice to check coverage against nor a rate to rank a candidate at. "
                "Run that campaign to `far` first"
                + ("" if index == 0 else ", or drop it from trials.sibling_configs")
            )
        intervals, livetime = analysed_intervals(arm_spec)
        model.add(
            SearchArm(
                key=arm_spec.arm,
                detectors=tuple(arm_spec.data.detectors),
                observing_run=str(arm_spec.data.observing_run),
                livetime_s=livetime,
                internal=True,
                far_curve_path=Path(curve),
                note=str(arm_spec.tag),
            ),
            ArmSegments(arm=arm_spec.arm, intervals=intervals, livetime_s=livetime),
        )
    return model


def run(spec, **kwargs) -> dict:
    """
    Stage driver: record the arms, and how much time each of them analysed.

    Runs after ``far`` and before ``candidates``, and deliberately does not touch a
    candidate table -- there is not one yet. What it produces is the *model*: which
    analyses were competing, and over exactly which seconds. ``candidates`` loads it and
    applies the correction, so the factor a candidate carries and the evidence for it are
    the same object.

    The model is written whatever the convention. Under ``none`` the factor is one for
    every candidate, and the record then says so explicitly rather than by the absence of
    a file -- which is also what a campaign that forgot to run this stage looks like.
    """
    siblings = [
        _load_sibling(name) for name in tuple(spec.trials.sibling_configs or ())
    ]
    model = build_model(spec, siblings)
    target = spec.path("trials", "trials_model.json")
    model.save(target)

    arms = {
        key: {
            "detectors": list(arm.detectors),
            "livetime_s": arm.livetime_s,
            "internal": arm.internal,
            "n_intervals": int(model.segments[key].intervals.shape[0]),
        }
        for key, arm in model.arms.items()
    }
    return {
        "model": str(target),
        "convention": model.convention,
        "arms": arms,
        "n_arms": len(model.arms),
        "n_internal_arms": len(model.internal_arms()),
        # The whole model: the arm list, the convention, and every analysed interval. The
        # intervals are the product -- they decide each candidate's factor one candidate
        # at a time -- so a summary of them is not what a downstream stage reads.
        "fingerprint": combine(
            model.convention,
            len(model.arms),
            digest_values(
                {
                    "convention": model.convention,
                    "fixed_factor": model.fixed_factor,
                    "arms": [
                        [
                            arm.key,
                            list(arm.detectors),
                            arm.observing_run,
                            arm.livetime_s,
                            arm.internal,
                        ]
                        for arm in model.arms.values()
                    ],
                    "intervals": {
                        key: segments.intervals
                        for key, segments in model.segments.items()
                    },
                }
            ),
        ),
    }


def _load_sibling(name: str):
    """
    Resolve one entry of ``trials.sibling_configs`` to a spec.

    Imported by the same loader the campaign itself came through, so a sibling named by
    path and one named by dotted module are the same arm -- which they are, since the
    spec hash no longer depends on the spelling.
    """
    from sage.search.spec import load_spec

    return load_spec(str(name))
