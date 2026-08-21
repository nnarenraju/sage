#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : candidates.py
Description   : The candidate table and the tiered inclusion criteria.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

This table is the single source of truth for every downstream table, figure and release
artefact; nothing downstream re-derives a candidate quantity.

Tiers follow the catalogue convention: a broad public candidate list, a confident subset,
and a further subset worth full parameter estimation.

Tiers are assigned twice. The search assigns them from significance and probability
alone, which is everything it can know without per-event work, and marks them
provisional. If candidates are later vetted, that verdict is recorded and the tiers are
re-derived, demoting anything the vetting rejected. A provisional tier is therefore an
upper bound on the final one, and the two are distinguishable so a provisional
classification is never mistaken for a vetted one.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TIER_UNDETERMINED: int = -1
TIER_CANDIDATE: int = 0
TIER_CONFIDENT: int = 1
TIER_PE: int = 2

CANDIDATE_COLUMNS: Tuple[str, ...] = (
    "name",
    "gps",
    "tc_gps",
    "stat",
    # Single-arm significance. These keep their names and their meaning whether or not a
    # trials correction has been applied, so nothing downstream changes silently.
    "far_per_yr",
    "ifar_yr",
    "far_hierarchical_per_yr",
    "p_value",
    # Trials-corrected significance, carried alongside rather than replacing the above.
    # See sage.search.trials: the factor is per candidate, because a network only had a
    # chance at times it actually analysed.
    "n_trials",
    "trials_convention",
    "covered_by",
    "found_by",
    "best_arm",
    "far_trials_per_yr",
    "ifar_trials_yr",
    "p_value_trials",
    "p_astro",
    "p_astro_lo",
    "p_astro_hi",
    "mchirp",
    "mchirp_sigma",
    "observing_run",
    "detectors",
    "tier",
    "tier_trials",
    "dq_vetoed",
    "dq_p_value",
    "cat1_clean",
    "catalogue_match",
    "catalogue_dt_s",
    "id_fraction",
    "is_new",
)


@dataclass
class CandidateTable:
    """The candidate list for one observing run."""

    columns: Dict[str, np.ndarray]
    attrs: Dict[str, object]

    def __len__(self) -> int:
        """Number of candidates."""
        if not self.columns:
            return 0
        return int(np.asarray(next(iter(self.columns.values()))).size)

    def tier(self, tier: int) -> "CandidateTable":
        """
        Subset at or above a tier; raises if any row is undetermined.

        An undetermined tier means the vetting that decides it has not run. Returning
        such a row inside a tier query would answer "is this candidate confident?" with a
        row that has not been asked the question, and the answer would be indistinguishable
        from a decided one.
        """
        tiers = np.asarray(self.columns["tier"], dtype=np.int64)
        undetermined = int(np.count_nonzero(tiers == TIER_UNDETERMINED))
        if undetermined:
            raise ValueError(
                f"{undetermined} of {len(self)} candidates carry tier "
                f"{TIER_UNDETERMINED} (undetermined), so a tier query cannot be answered "
                "for them. Run apply_tiers, or retier with the vetting verdict"
            )
        return self._select(tiers >= int(tier))

    def new_events(self) -> "CandidateTable":
        """
        Candidates with no catalogue counterpart.

        Read from ``is_new``, which the crossmatch writes. A table that has not been
        crossmatched cannot answer this: every candidate would look new, which is the
        most interesting possible answer and the one most likely to be wrong.
        """
        if "is_new" not in self.columns:
            raise ValueError(
                "this table carries no 'is_new' column, so it has not been crossmatched "
                "against the published catalogues. Every candidate would report as new"
            )
        return self._select(np.asarray(self.columns["is_new"], dtype=bool))

    def _select(self, mask: np.ndarray) -> "CandidateTable":
        """The rows where ``mask`` holds, with the attributes carried through."""
        mask = np.asarray(mask, dtype=bool)
        return CandidateTable(
            columns={
                name: np.asarray(values)[mask]
                for name, values in self.columns.items()
            },
            attrs=dict(self.attrs),
        )

    def save(self, path: str | Path) -> None:
        """
        Write ``candidates.h5``.

        Refuses a table holding a non-nan mass or spin whose provenance is not ``pe``.
        Sage's heads estimate ``tc`` and ``mchirp`` and nothing else, so any other mass or
        spin column can only have come from parameter estimation -- and a follow-up
        template's parameters written into one would be indistinguishable from a
        measurement.
        """
        from sage.utils.atomic_io import atomic_h5

        self._check_parameter_provenance()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with atomic_h5(target, mode="w") as handle:
            for key, value in (self.attrs or {}).items():
                handle.attrs[key] = value
            handle.attrs["columns"] = list(self.columns)
            for name, values in self.columns.items():
                values = np.asarray(values)
                if values.dtype.kind in "SUO":
                    values = np.asarray(
                        [str(v) for v in values], dtype=h5_string_dtype()
                    )
                handle.create_dataset(name, data=values)

    def _check_parameter_provenance(self) -> None:
        """Guard the no-fabricated-parameters rule at the point of writing."""
        provenance = self.columns.get("mass_provenance")
        for name in ("mass1", "mass2", "spin1z", "spin2z"):
            values = self.columns.get(name)
            if values is None:
                continue
            filled = np.isfinite(np.asarray(values, dtype=np.float64))
            if not filled.any():
                continue
            if provenance is None:
                raise ValueError(
                    f"{name} holds {int(filled.sum())} finite values but the table "
                    "records no mass_provenance. Sage estimates tc and mchirp only, so "
                    "these can only have come from parameter estimation and must say so"
                )
            bad = filled & (np.asarray(provenance).astype(str) != "pe")
            if bad.any():
                raise ValueError(
                    f"{name} holds {int(bad.sum())} finite values whose provenance is "
                    "not 'pe'; a follow-up template's parameters are not a measurement"
                )

    @classmethod
    def load(cls, path: str | Path, allow_undetermined: bool = False) -> "CandidateTable":
        """
        Read the candidate table.

        ``allow_undetermined`` is off by default, so a table still carrying provisional
        tiers is refused where a decided one was expected. The release stage reads it
        that way deliberately: publishing a list whose tiers were never decided is the
        failure the flag exists to prevent.
        """
        import h5py

        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"no candidate table at {target}")
        with h5py.File(target, "r") as handle:
            if "columns" not in handle.attrs:
                raise ValueError(
                    f"{target} records no column list, so which columns it holds cannot "
                    "be established"
                )
            names = [
                v.decode() if isinstance(v, bytes) else str(v)
                for v in handle.attrs["columns"]
            ]
            columns = {}
            for name in names:
                if name not in handle:
                    raise ValueError(
                        f"{target} declares column {name!r} and does not hold it; the "
                        "file was truncated part-way through a write"
                    )
                values = handle[name][()]
                if values.dtype.kind in "SO":
                    values = np.asarray(
                        [v.decode() if isinstance(v, bytes) else str(v) for v in values]
                    )
                columns[name] = values
            table = cls(
                columns=columns,
                attrs={k: handle.attrs[k] for k in handle.attrs if k != "columns"},
            )
        if not allow_undetermined and "tier" in table.columns:
            undetermined = int(
                np.count_nonzero(
                    np.asarray(table.columns["tier"], dtype=np.int64)
                    == TIER_UNDETERMINED
                )
            )
            if undetermined:
                raise ValueError(
                    f"{target} holds {undetermined} candidates with an undetermined "
                    "tier. Pass allow_undetermined=True to read a provisional list"
                )
        return table


def h5_string_dtype():
    """Variable-length UTF-8 string dtype, for the text columns."""
    import h5py

    return h5py.string_dtype(encoding="utf-8")


def from_triggers(
    clustered: "object",
    far_curve,
    pastro_table: Optional[object] = None,
    spec=None,
) -> CandidateTable:
    """
    Assemble candidates from clustered zero-lag triggers, FAR and p_astro.

    The triggers must be clustered. One physical event spans many analysis windows, and
    an unclustered list would enter the same event repeatedly -- each copy with its own
    name and its own rate, and the repetition invisible in the table.

    Every quantity is read from the object that owns it: rates from the FAR curve,
    probabilities from the p_astro table, times and statistics from the triggers. Nothing
    is recomputed here, so a candidate's rate is the same number the FAR stage published
    and cannot drift from it.

    p_astro is joined **by time**, not by row order. The two tables are produced by
    different stages from different intermediate products, and matching them positionally
    would silently pair candidate *i* with probability *j* the moment either changed its
    ordering or its length.

    Mass and spin columns are deliberately absent. Sage's heads estimate ``tc`` and
    ``mchirp``; anything else belongs to parameter estimation and is added by the
    follow-up track with its provenance.
    """
    from sage.search.naming import disambiguate, name_from_gps

    columns = getattr(clustered, "columns", clustered)
    if not bool(getattr(clustered, "attrs", {}).get("clustered", False)):
        raise ValueError(
            "from_triggers needs a clustered trigger set: one event spans many analysis "
            "windows, so an unclustered list would enter the same event once per window"
        )
    gps = np.asarray(columns["gps"], dtype=np.float64)
    stat = np.asarray(columns["stat"], dtype=np.float64)
    order = np.argsort(-stat)
    gps, stat = gps[order], stat[order]

    out: Dict[str, np.ndarray] = {
        # Disambiguated, because the stamp names a second and clustering only separates
        # candidates by 0.35 s. The name is the identity every later join uses, so a
        # collision makes those joins ambiguous rather than merely untidy.
        "name": np.asarray(
            disambiguate([name_from_gps(float(t)) for t in gps], gps)
        ),
        "gps": gps,
        "stat": stat,
        "far_per_yr": np.asarray(far_curve.far_of(stat), dtype=np.float64),
        "ifar_yr": np.asarray(far_curve.ifar_of(stat), dtype=np.float64),
    }
    for source, target in (("tc_gps", "tc_gps"), ("mchirp", "mchirp"),
                           ("mchirp_sigma", "mchirp_sigma")):
        if source in columns:
            out[target] = np.asarray(columns[source], dtype=np.float64)[order]

    # p-value over the foreground the curve was read against, which is the exposure the
    # analysis that produced these candidates actually ran for.
    observation_yr = float(far_curve.foreground_livetime_s) / SECONDS_PER_JULIAN_YEAR
    out["p_value"] = -np.expm1(-out["far_per_yr"] * observation_yr)

    if pastro_table is not None:
        out.update(_join_pastro(gps, stat, pastro_table))

    if spec is not None:
        out["observing_run"] = np.asarray(
            [str(spec.data.observing_run)] * gps.size
        )
        out["detectors"] = np.asarray(
            [",".join(spec.data.detectors)] * gps.size
        )
    out["tier"] = np.full(gps.size, TIER_UNDETERMINED, dtype=np.int64)

    attrs = {"clustered": True, "provisional_tiers": True}
    if spec is not None:
        attrs.update(
            observing_run=str(spec.data.observing_run),
            arm=str(spec.arm),
            spec_hash=str(spec.hash()),
        )
    return CandidateTable(columns=out, attrs=attrs)


#: Seconds in a Julian year, matching the convention every rate here is quoted in.
SECONDS_PER_JULIAN_YEAR: float = 365.25 * 86400.0

#: How close a candidate and a p_astro row must be to be the same event. The p_astro table
#: is built from the same clustered triggers, so the times are equal to the last bit in
#: the ordinary case; the window exists so that a table rebuilt through a different float
#: path still joins, and is far tighter than the clustering window that separates events.
PASTRO_JOIN_S: float = 1.0e-3


def _join_pastro(gps: np.ndarray, stats: np.ndarray, pastro_table) -> Dict[str, np.ndarray]:
    """
    Match each candidate to its p_astro row by time.

    An unmatched candidate is an error rather than a nan. Both tables come from the same
    clustered zero-lag set, so a candidate with no probability means the two were built
    from different trigger sets -- and a nan there would read as "not astrophysical"
    everywhere it is used.
    """
    theirs = np.asarray(pastro_table.gps, dtype=np.float64)
    probability = pastro_table.astrophysical()
    order = np.argsort(theirs)
    sorted_times = theirs[order]
    nearest = np.clip(np.searchsorted(sorted_times, gps), 0, sorted_times.size - 1)
    left = np.clip(nearest - 1, 0, sorted_times.size - 1)
    pick = np.where(
        np.abs(sorted_times[nearest] - gps) <= np.abs(sorted_times[left] - gps),
        nearest,
        left,
    )
    matched = order[pick]
    gap = np.abs(theirs[matched] - gps)

    # p_astro is fitted on a bounded support, and the monotonicity policy may narrow it
    # further, so the mixture legitimately scores only part of the trigger set. A
    # candidate outside what was scored has no probability -- that is a fact about the
    # fit, not a mismatch -- and is recorded as nan, which every consumer already treats
    # as "no astrophysical probability" and which apply_tiers refuses to promote on.
    scored = np.asarray(pastro_table.stat, dtype=np.float64)
    stat = np.asarray(stats, dtype=np.float64)
    inside = (stat >= scored.min()) & (stat <= scored.max())

    # Inside the scored range, a missing row IS a mismatch: both tables come from the same
    # clustered zero-lag set, so a candidate the mixture should have seen and did not
    # means they were built from different trigger sets.
    missing = inside & (gap > PASTRO_JOIN_S)
    if np.any(missing):
        worst = int(np.argmax(np.where(missing, gap, -np.inf)))
        raise ValueError(
            f"{int(np.count_nonzero(missing))} of {int(inside.sum())} candidates inside "
            f"the p_astro support have no row within {PASTRO_JOIN_S} s; the worst is "
            f"{gap[worst]:.6g} s at GPS {gps[worst]:.6f}. The two tables were built from "
            "different trigger sets"
        )

    signal, _ = _astro_bounds(pastro_table)
    take = inside & (gap <= PASTRO_JOIN_S)

    def _gather(values):
        out = np.full(gps.size, np.nan, dtype=np.float64)
        out[take] = np.asarray(values, dtype=np.float64)[matched][take]
        return out

    return {
        "p_astro": _gather(probability),
        "p_astro_lo": _gather(signal[0]),
        "p_astro_hi": _gather(signal[1]),
        "p_astro_scored": take,
    }


def _astro_bounds(pastro_table):
    """The astrophysical component's credible bounds, summed the way the value is."""
    from sage.search.pastro.categories import DEFAULT_CATEGORIES

    astro = {c.name for c in DEFAULT_CATEGORIES if c.astrophysical}
    present = [name for name in pastro_table.probabilities if name in astro]
    lower = np.sum([pastro_table.lower[name] for name in present], axis=0)
    upper = np.sum([pastro_table.upper[name] for name in present], axis=0)
    return (lower, upper), present


def apply_tiers(
    table: CandidateTable,
    far_candidate_per_day: float = 2.0,
    p_astro_confident: float = 0.5,
    far_pe_per_yr: float = 1.0,
    use_dataquality: bool = False,
) -> CandidateTable:
    """
    Assign tiers from significance and probability.

    Thresholds are strict inequalities so a candidate sitting exactly on a boundary is
    excluded rather than admitted.

    Both tier columns are written whenever the trials columns are present: ``tier`` from
    the single-arm rates and ``tier_trials`` from the corrected ones. They are kept
    separate rather than reconciled, because a candidate that qualifies under one and not
    the other is exactly what a reader needs to see.

    Parameters
    ----------
    use_dataquality : bool
        When false the result is provisional: no vetting verdict is required, and every
        row is marked as such. When true a verdict must be present for each candidate,
        and one that fails it is demoted regardless of its significance.
    """
    from sage.search.trials import assign_tiers

    far = np.asarray(table.columns["far_per_yr"], dtype=np.float64)
    p_astro = (
        np.asarray(table.columns["p_astro"], dtype=np.float64)
        if "p_astro" in table.columns
        else None
    )
    columns = dict(table.columns)
    # One rule, owned by trials.assign_tiers, applied to whichever rate is being tiered.
    tiers = assign_tiers(far, p_astro, p_astro_confident, far_pe_per_yr)

    # The broad list is bounded by rate, in the units the threshold is quoted in. Rows
    # above it are dropped rather than given a tier: the ladder starts at
    # TIER_CANDIDATE ("in the public list") and has no rung below it, and reusing
    # TIER_UNDETERMINED would say "not yet decided" about a row that has been decided.
    candidate_far = float(far_candidate_per_day) * DAYS_PER_JULIAN_YEAR
    keep = far < candidate_far

    tiers = tiers[keep]
    far = far[keep]
    if p_astro is not None:
        p_astro = p_astro[keep]
    columns = {name: np.asarray(values)[keep] for name, values in columns.items()}

    if use_dataquality:
        if "dq_vetoed" not in columns:
            raise ValueError(
                "use_dataquality=True but the table carries no 'dq_vetoed' column; the "
                "vetting verdict has not been recorded, and treating its absence as a "
                "pass would publish a vetted tier that was never vetted"
            )
        vetoed = np.asarray(columns["dq_vetoed"], dtype=bool)
        tiers = np.where(vetoed, TIER_CANDIDATE, tiers)
    columns["tier"] = tiers.astype(np.int64)

    if "far_trials_per_yr" in columns:
        columns["tier_trials"] = assign_tiers(
            np.asarray(columns["far_trials_per_yr"], dtype=np.float64),
            p_astro,
            p_astro_confident,
            far_pe_per_yr,
        ).astype(np.int64)

    attrs = dict(table.attrs)
    attrs["provisional_tiers"] = not bool(use_dataquality)
    attrs["n_below_threshold"] = int(np.count_nonzero(~keep))
    attrs["far_candidate_per_day"] = float(far_candidate_per_day)
    attrs["p_astro_confident"] = float(p_astro_confident)
    attrs["far_pe_per_yr"] = float(far_pe_per_yr)
    return CandidateTable(columns=columns, attrs=attrs)


#: Days in a Julian year, for converting a per-day threshold to the per-year rates the
#: FAR curve is quoted in.
DAYS_PER_JULIAN_YEAR: float = 365.25


def retier(table: CandidateTable, dq_reports) -> CandidateTable:
    """
    Re-derive tiers using vetting verdicts gathered after the search.

    Only demotes: vetting can reject a candidate but never promote one, so a tier can
    fall when a verdict arrives and never rise.

    The demotion is enforced by taking the elementwise minimum against the tiers already
    recorded, rather than by trusting the re-derivation to be monotone. A verdict that
    somehow raised a tier would otherwise publish a candidate as more confident *because*
    it was vetted, which is the one direction vetting must never move a result.

    A candidate with no verdict keeps its provisional tier and stays marked provisional.
    Treating a missing verdict as a pass is how an unvetted candidate reaches a vetted
    list.
    """
    verdicts = _verdicts_by_name(table, dq_reports)
    columns = dict(table.columns)
    columns["dq_vetoed"] = verdicts["vetoed"]
    if "p_value" in verdicts:
        columns["dq_p_value"] = verdicts["p_value"]

    decided = verdicts["decided"]
    before = np.asarray(columns["tier"], dtype=np.int64)
    revised = apply_tiers(
        CandidateTable(columns=columns, attrs=dict(table.attrs)),
        far_candidate_per_day=float(
            table.attrs.get("far_candidate_per_day", 2.0)
        ),
        p_astro_confident=float(table.attrs.get("p_astro_confident", 0.5)),
        far_pe_per_yr=float(table.attrs.get("far_pe_per_yr", 1.0)),
        use_dataquality=True,
    )
    tiers = np.minimum(np.asarray(revised.columns["tier"], dtype=np.int64), before)
    # Undecided candidates keep exactly what they had.
    tiers = np.where(decided, tiers, before)
    out = dict(revised.columns)
    out["tier"] = tiers
    attrs = dict(revised.attrs)
    attrs["provisional_tiers"] = bool(np.any(~decided))
    attrs["n_vetted"] = int(np.count_nonzero(decided))
    return CandidateTable(columns=out, attrs=attrs)


def _verdicts_by_name(table: CandidateTable, dq_reports) -> Dict[str, np.ndarray]:
    """
    Align vetting verdicts to the candidate rows by name.

    By name, not by position: the verdicts arrive from a per-candidate follow-up that may
    have run on a subset, in whatever order the jobs finished.
    """
    names = [str(name) for name in table.columns["name"]]
    lookup = {
        str(getattr(report, "name", report.get("name"))): report
        for report in (dq_reports or [])
    }
    vetoed = np.zeros(len(names), dtype=bool)
    decided = np.zeros(len(names), dtype=bool)
    p_value = np.full(len(names), np.nan, dtype=np.float64)
    for index, name in enumerate(names):
        report = lookup.get(name)
        if report is None:
            continue
        decided[index] = True
        vetoed[index] = bool(
            getattr(report, "vetoed", None)
            if hasattr(report, "vetoed")
            else report.get("vetoed", False)
        )
        value = (
            getattr(report, "p_value", None)
            if hasattr(report, "p_value")
            else report.get("p_value")
        )
        if value is not None:
            p_value[index] = float(value)
    return {"vetoed": vetoed, "decided": decided, "p_value": p_value}


def recompute_far(table: CandidateTable, far_curve) -> CandidateTable:
    """
    Re-assign FAR after a change to the background, without re-running inference.

    The tiers are reset to undetermined. They were derived from the old rates, and a table
    carrying new rates beside tiers computed from the old ones is internally inconsistent
    in a way nothing later would detect -- :meth:`CandidateTable.tier` would answer from
    them without complaint. Re-run :func:`apply_tiers`.

    ``p_astro`` is untouched. It is a posterior from a rate mixture fitted to a whole
    trigger set, not a function of one candidate's rate, so it cannot be updated by
    re-reading a curve; refitting is p_astro's own stage.
    """
    stat = np.asarray(table.columns["stat"], dtype=np.float64)
    columns = dict(table.columns)
    columns["far_per_yr"] = np.asarray(far_curve.far_of(stat), dtype=np.float64)
    columns["ifar_yr"] = np.asarray(far_curve.ifar_of(stat), dtype=np.float64)
    observation_yr = float(far_curve.foreground_livetime_s) / SECONDS_PER_JULIAN_YEAR
    columns["p_value"] = -np.expm1(-columns["far_per_yr"] * observation_yr)
    columns["tier"] = np.full(stat.size, TIER_UNDETERMINED, dtype=np.int64)
    attrs = dict(table.attrs)
    attrs["provisional_tiers"] = True
    return CandidateTable(columns=columns, attrs=attrs)


def expected_contamination(table: CandidateTable) -> dict:
    """
    Expected number of noise events in a candidate set.

    The sum of ``1 - p_astro`` over a set is its expected terrestrial count, which is
    reported alongside any detection claim.

    Reported per tier as well as overall, because that is how the claim is made: "N
    confident candidates, of which an expected M are terrestrial" is a statement about the
    confident subset, and the number for the broad list does not bound it.
    """
    if "p_astro" not in table.columns:
        raise ValueError(
            "this table carries no p_astro, so its expected terrestrial count is not "
            "defined. Run the pastro stage first"
        )
    probability = np.asarray(table.columns["p_astro"], dtype=np.float64)
    tiers = np.asarray(table.columns["tier"], dtype=np.int64)
    out = {
        "n_candidates": int(probability.size),
        "expected_terrestrial": float(np.sum(1.0 - probability)),
        "expected_astrophysical": float(np.sum(probability)),
    }
    for name, tier in (
        ("candidate", TIER_CANDIDATE),
        ("confident", TIER_CONFIDENT),
        ("pe", TIER_PE),
    ):
        mask = tiers >= tier
        out[f"n_{name}"] = int(np.count_nonzero(mask))
        out[f"expected_terrestrial_{name}"] = float(
            np.sum(1.0 - probability[mask])
        )
    return out


def run(spec, **kwargs) -> dict:
    """
    Stage driver: assemble the campaign's candidate list.

    Reads the clustered zero-lag triggers, the counted FAR curve, the p_astro table and
    the trials model, and writes one table that every downstream figure, table and release
    artefact reads. Nothing downstream re-derives a candidate quantity, which is what
    keeps a number in a figure and the same number in a table from drifting apart.

    Tiers are provisional here and say so. Deciding them needs per-candidate vetting,
    which is the follow-up track's `retier`; publishing a list whose tiers were never
    decided is what ``CandidateTable.load`` refuses by default.
    """
    from sage.search.background import cluster_zerolag
    from sage.search.far import FarCurve
    from sage.search.fingerprint import combine, digest_h5
    from sage.search.pastro.assign import PAstroTable
    from sage.search.trials import TrialsModel, apply as apply_trials, build_records
    from sage.search.triggers import read_shard

    zerolag, _ = read_shard(spec.path("zerolag", "zerolag_slide0000.h5"))
    clustered = cluster_zerolag(
        zerolag,
        window_s=float(spec.cluster.window_s),
        linkage=spec.cluster.linkage,
    )
    curve = FarCurve.load(
        spec.path("far", f"far_curve_{spec.data.observing_run}_inclusive.h5")
    )
    pastro = PAstroTable.load(spec.path("pastro", "pastro_table.h5"))

    table = from_triggers(clustered, curve, pastro_table=pastro, spec=spec)

    # The trials correction is applied before tiering, so `tier_trials` exists to be
    # written alongside `tier` in the same pass.
    model = TrialsModel.load(spec.path("trials", "trials_model.json"))
    # This campaign's own clustered triggers are handed in as its arm's, so `found_by`
    # and therefore `best_arm` are recorded. Without them the factor is still computable
    # -- it counts arms that *could* have produced a false alarm, not arms that did -- but
    # the corrected p-value has no arm to take an observation time from, and `apply`
    # refuses rather than picking one. A sibling arm's triggers are added here when a
    # multi-arm campaign is combined.
    records = build_records(
        table,
        model,
        triggers_by_arm={str(spec.arm): clustered},
        match_window_s=float(spec.trials.match_window_s),
    )
    # The table itself, not its columns: `apply` rebuilds the caller's own type and
    # carries the attributes across, where a bare dict comes back as a bare dict with the
    # provenance dropped.
    table = apply_trials(table, records, model)
    table = apply_tiers(
        table,
        far_candidate_per_day=float(spec.significance.candidate_far_per_day),
        p_astro_confident=float(spec.pastro.confident_p_astro),
        far_pe_per_yr=float(spec.pastro.pe_far_per_yr),
    )

    target = spec.path("candidates", "candidates.h5")
    table.save(target)
    contamination = expected_contamination(table)
    return {
        "table": str(target),
        "n_candidates": len(table),
        "n_below_threshold": int(table.attrs.get("n_below_threshold", 0)),
        "provisional_tiers": bool(table.attrs.get("provisional_tiers", True)),
        "contamination": contamination,
        "loudest": (
            {
                "name": str(table.columns["name"][0]),
                "stat": float(table.columns["stat"][0]),
                "ifar_yr": float(table.columns["ifar_yr"][0]),
                "p_astro": float(table.columns["p_astro"][0]),
            }
            if len(table)
            else None
        ),
        "fingerprint": combine(len(table), digest_h5(target)),
    }
