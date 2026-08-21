#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_trials.py
Description   : The trials factor, its per-candidate coverage, and the two views.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The factor is a scalar, so the risk is not in the arithmetic. It is in counting the
wrong thing: applying a campaign-wide constant where coverage varies, counting analyses
that found a candidate instead of analyses that could have, or overwriting the
uncorrected numbers so the correction can never be inspected or undone.

Runs on synthetic arms and segments; needs no data, no GPU and no network.
"""

import numpy as np
import pytest

from sage.search.background import BackgroundSet
from sage.search.far import build_far_curve
from sage.search.geometry import SearchGeometry
from sage.search.grid import AnalysisGrid
from sage.search.segments import Segment, coincident_intervals
from sage.search.trials import (
    CONVENTIONS,
    TIER_CONFIDENT,
    TIER_PE,
    ArmSegments,
    SearchArm,
    TrialsModel,
    TrialsRecord,
    apply,
    assign_best_arm,
    build_records,
    comparison,
    summary,
    trials_factor,
    without_trials,
)

# Start of O3a, so the times carry the float64 resolution the real ones do.
T0 = 1238166018.0

# The O3a livetimes the per-candidate factor exists for: 26 days of HL time that the HLV
# arm never analysed, and so never had a chance at.
HL_LIVETIME_S = 106.75 * 86400.0
HLV_LIVETIME_S = 80.81 * 86400.0

RATE = 2048.0
CHUNK_S = 512.0
OVERLAP_S = 15.5994

GEOMETRY = SearchGeometry(
    sample_rate=RATE,
    signal_length_s=12.0,
    padding_length_s=2.0,
    stride_samples=205,
    tc_lower_s=5.0,
    tc_upper_s=7.0,
)


def _segments(detector, n_chunks=3):
    """Overlapping chunks in the release's own layout, for one detector."""
    step = CHUNK_S - OVERLAP_S
    nsamples = int(round(CHUNK_S * RATE))
    return [
        Segment(
            segment_index=k,
            detector=detector,
            observing_run="O3a",
            gps_start=T0 + k * step,
            gps_end=T0 + k * step + CHUNK_S,
            sample_rate=RATE,
            nsamples=nsamples,
            sample_start_idx=k * nsamples,
            dyn_range_fac=1.0,
            noise_low_freq_cutoff=15.0,
        )
        for k in range(n_chunks)
    ]


def _intervals(*spans):
    """Analysed intervals as the (n, 2) array the model stores."""
    return np.asarray(spans, dtype=np.float64).reshape(-1, 2)


def _model(
    convention="coverage", fixed_factor=None, external=False, arms=("hl", "hlv")
):
    """
    HL over the whole hour, HLV over 200 s inside it, as O3a is in miniature.

    The asymmetry is the point: a candidate outside the triple-coincident stretch was
    reachable by one arm only.
    """
    model = TrialsModel(convention=convention, fixed_factor=fixed_factor)
    if "hl" in arms:
        model.add(
            SearchArm(
                key="hl",
                detectors=("H1", "L1"),
                observing_run="O3a",
                livetime_s=HL_LIVETIME_S,
            ),
            ArmSegments(arm="hl", intervals=_intervals((T0, T0 + 1000.0))),
        )
    if "lv" in arms:
        model.add(
            SearchArm(
                key="lv",
                detectors=("L1", "V1"),
                observing_run="O3a",
                livetime_s=HLV_LIVETIME_S,
            ),
            ArmSegments(arm="lv", intervals=_intervals((T0 + 300.0, T0 + 700.0))),
        )
    if "hlv" in arms:
        model.add(
            SearchArm(
                key="hlv",
                detectors=("H1", "L1", "V1"),
                observing_run="O3a",
                livetime_s=HLV_LIVETIME_S,
            ),
            ArmSegments(arm="hlv", intervals=_intervals((T0 + 400.0, T0 + 600.0))),
        )
    if external:
        model.add(
            SearchArm(
                key="gwtc3",
                detectors=("H1", "L1", "V1"),
                observing_run="O3a",
                internal=False,
                note="another group's analysis of the same data",
            ),
            ArmSegments(arm="gwtc3", intervals=_intervals((T0, T0 + 1000.0))),
        )
    return model


def _candidates(names, gps, far_per_yr, **columns):
    """A candidate table as the columnar mapping the trials stage accepts."""
    table = {
        "name": np.asarray(names),
        "gps": np.asarray(gps, dtype=np.float64),
        "far_per_yr": np.asarray(far_per_yr, dtype=np.float64),
    }
    table["ifar_yr"] = columns.pop("ifar_yr", 1.0 / table["far_per_yr"])
    for key, values in columns.items():
        table[key] = np.asarray(values)
    return table


def _triggers(gps, stat):
    """One arm's clustered triggers, in the shard schema's column names."""
    return {
        "gps": np.asarray(gps, dtype=np.float64),
        "stat": np.asarray(stat, dtype=np.float64),
    }


def _curve(background_stats, livetime_s=1.0e6):
    """A FAR curve from a stated background, so an arm's ranking is a measured one."""
    return build_far_curve(
        BackgroundSet(
            stats=np.asarray(background_stats, dtype=np.float64),
            livetime_s=livetime_s,
            n_slides=100,
            removal="inclusive",
        ),
        foreground_livetime_s=HL_LIVETIME_S,
    )


class TestCoverage:
    """Which arms had a chance, established from analysed time alone."""

    def test_coverage_counts_analysing_arms(self):
        """
        A candidate in two-detector-only time is covered by the two-detector arm alone.

        This is the whole reason the factor is per candidate. A campaign-wide constant
        would penalise every candidate outside the most restrictive network's livetime
        for chances that network never had.

        Asserting the tuples and not only the counts: a factor of one is also what a
        wrong implementation returns when it finds no coverage at all, and the two are
        told apart by which arm is named.
        """
        model = _model()
        table = _candidates(
            ["outside", "inside"], [T0 + 100.0, T0 + 500.0], [1.0, 1.0]
        )
        records = build_records(table, model)

        assert records[0].covered_by == ("hl",)
        assert records[1].covered_by == ("hl", "hlv")
        assert [record.n_trials for record in records] == [1, 2]

    def test_coverage_uses_analysed_time(self):
        """
        Coverage comes from the window lattice, not the observing segments.

        A window needs a whole window of contiguous data, so analysed time is strictly
        less than coincident time; using the larger one would credit an arm with a
        chance at a moment it could not have produced a trigger for.

        The negative control is the same time tested against the coincident intervals,
        which does report coverage there -- so this fails for an implementation that
        reads the observing segments rather than passing by luck.
        """
        segments = {"H1": _segments("H1"), "L1": _segments("L1")}
        coincident = coincident_intervals(segments)
        grid = AnalysisGrid.build(GEOMETRY, segments, coincident)
        analysed = ArmSegments.from_grid("hl", grid)
        observing = ArmSegments(arm="hl", intervals=_intervals(*coincident))

        # The exact analysed time is an integer window count times the stride.
        assert analysed.livetime_s == grid.livetime_s
        assert 0.90 < analysed.livetime_s / observing.livetime_s < 1.0

        # Inside the last chunk, but within one window of its end: data exists and no
        # window start can sit there.
        last_end = max(segment.gps_end for segment in segments["H1"])
        late = last_end - 1.0
        assert bool(observing.contains(np.array([late]))[0])
        assert not bool(analysed.contains(np.array([late]))[0])

        # And a time the lattice did reach is still covered, so the test is not passing
        # by reporting nothing at all.
        first_start = float(grid.reference_spans[0].first_gps)
        assert bool(analysed.contains(np.array([first_start]))[0])

    def test_found_implies_covered(self):
        """An arm that produced a trigger must be recorded as covering that time."""
        model = _model()
        table = _candidates(["c"], [T0 + 500.0], [1.0])
        records = build_records(
            table,
            model,
            triggers_by_arm={
                "hl": _triggers([T0 + 500.02], [11.0]),
                "hlv": _triggers([T0 + 499.97], [10.0]),
            },
        )
        assert set(records[0].found_by) <= set(records[0].covered_by)

        # A trigger from an arm whose analysed segments do not reach that time is a
        # segments error, and is raised rather than quietly widening the coverage.
        outside = _candidates(["c"], [T0 + 100.0], [1.0])
        with pytest.raises(ValueError, match="analysed"):
            build_records(
                outside, model, triggers_by_arm={"hlv": _triggers([T0 + 100.0], [10.0])}
            )

        # The same invariant holds for a hand-built record.
        with pytest.raises(ValueError, match="not recorded as covering"):
            TrialsRecord(
                candidate="c", gps=T0, covered_by=("hl",), found_by=("hl", "hlv")
            )

    def test_zero_coverage_is_rejected(self):
        """A candidate covered by no arm indicates wrong segments, and raises."""
        model = _model()
        table = _candidates(["c"], [T0 + 2000.0], [1.0])
        with pytest.raises(ValueError, match="analysed time"):
            build_records(table, model)

        with pytest.raises(ValueError, match="no internal arm"):
            trials_factor(TrialsRecord(candidate="c", gps=T0 + 2000.0), model)

    def test_external_catalogues_do_not_count(self):
        """
        Another group re-analysing the data does not inflate this search's factor.

        Their coverage is recorded, because it bears on whether a candidate is new, but
        it says nothing about how often this pipeline produces a false alarm.

        The control is the same campaign with that arm marked internal, which does give
        three: the arm is genuinely covering the time, so only the ``internal`` flag can
        be producing the difference.
        """
        table = _candidates(["c"], [T0 + 500.0], [1.0])
        records = build_records(table, _model(external=True))
        assert records[0].covered_by == ("hl", "hlv", "gwtc3")
        assert records[0].n_trials == 2

        internalised = _model()
        internalised.add(
            SearchArm(key="gwtc3", detectors=("H1", "L1", "V1"), observing_run="O3a"),
            ArmSegments(arm="gwtc3", intervals=_intervals((T0, T0 + 1000.0))),
        )
        assert build_records(table, internalised)[0].n_trials == 3


class TestFactor:
    """The scalar itself, under each convention."""

    def test_convention_counts_covering(self):
        """
        Under the default convention the factor is the number of covering arms.

        Counted whether or not the arm found anything: the candidate here is found by
        one arm and covered by two, so an implementation counting detections returns one
        and fails.
        """
        model = _model()
        table = _candidates(["c"], [T0 + 500.0], [1.0])
        records = build_records(
            table, model, triggers_by_arm={"hl": _triggers([T0 + 500.0], [11.0])}
        )
        assert records[0].found_by == ("hl",)
        assert records[0].covered_by == ("hl", "hlv")
        assert trials_factor(records[0], model) == 2

    def test_convention_counts_finding(self):
        """Under the detection convention only arms that produced a trigger count."""
        model = _model(convention="detection")
        table = _candidates(["one", "both"], [T0 + 500.0, T0 + 550.0], [1.0, 1.0])
        records = build_records(
            table,
            model,
            triggers_by_arm={
                "hl": _triggers([T0 + 500.0, T0 + 550.0], [11.0, 12.0]),
                "hlv": _triggers([T0 + 550.04], [10.0]),
            },
        )
        # Both candidates sit in triple-coincident time, so coverage alone cannot tell
        # them apart; only the detections can.
        assert records[0].covered_by == records[1].covered_by == ("hl", "hlv")
        assert [trials_factor(record, model) for record in records] == [1, 2]

    def test_none_convention_is_exactly_one(self):
        """The uncorrected view is the corrected one with a factor of one."""
        model = _model(convention="none")
        table = _candidates(["c"], [T0 + 500.0], [3.0])
        records = build_records(table, model)
        assert records[0].covered_by == ("hl", "hlv")
        assert trials_factor(records[0], model) == 1

        corrected = apply(table, records, model, observation_time_s=HL_LIVETIME_S)
        assert corrected["far_trials_per_yr"][0] == table["far_per_yr"][0]
        assert corrected["ifar_trials_yr"][0] == table["ifar_yr"][0]

    def test_factor_is_at_least_one(self):
        """No convention can produce a factor below one."""
        table = _candidates(["c"], [T0 + 500.0], [1.0])
        triggers = {"hl": _triggers([T0 + 500.0], [11.0])}
        for convention in CONVENTIONS:
            model = _model(convention=convention, fixed_factor=1)
            record = build_records(table, model, triggers_by_arm=triggers)[0]
            assert trials_factor(record, model) >= 1, convention

        # A stated factor below one is refused rather than clamped, so a mis-set model
        # cannot quietly weaken every rate it touches.
        zeroed = _model(convention="fixed", fixed_factor=0)
        with pytest.raises(ValueError, match="at least one"):
            trials_factor(
                TrialsRecord(candidate="c", gps=T0 + 500.0, covered_by=("hl",)), zeroed
            )

    def test_single_arm_unchanged(self):
        """One arm leaves the rates identical, not merely close."""
        model = _model(arms=("hl",))
        table = _candidates(
            ["c"], [T0 + 500.0], [0.7], p_astro=[0.9]
        )
        records = build_records(
            table, model, triggers_by_arm={"hl": _triggers([T0 + 500.0], [11.0])}
        )
        corrected = apply(table, records, model)

        assert corrected["n_trials"].tolist() == [1]
        assert np.array_equal(corrected["far_trials_per_yr"], table["far_per_yr"])
        assert np.array_equal(corrected["ifar_trials_yr"], table["ifar_yr"])
        assert np.array_equal(corrected["p_value_trials"], corrected["p_value"])


class TestApplication:
    """What the correction does to the table."""

    def test_uncorrected_columns_are_untouched(self):
        """
        Applying the correction does not modify far_per_yr or ifar_yr.

        The correction must stay reversible and inspectable; a pipeline that overwrites
        the single-arm rate cannot show a reader what the factor did. Checked against
        copies taken before the call, so an in-place multiplication is caught even
        though the returned table would look right.
        """
        model = _model()
        table = _candidates(["c"], [T0 + 500.0], [2.0], p_astro=[0.9])
        before = {key: values.copy() for key, values in table.items()}
        records = build_records(
            table, model, triggers_by_arm={"hl": _triggers([T0 + 500.0], [11.0])}
        )
        corrected = apply(table, records, model)

        assert corrected["far_trials_per_yr"][0] != table["far_per_yr"][0]
        for key, values in before.items():
            assert np.array_equal(table[key], values), key
            assert np.array_equal(corrected[key], values), key
        assert "far_trials_per_yr" not in table

    def test_corrected_far_is_the_product(self):
        """
        far_trials_per_yr equals n_trials times far_per_yr, exactly.

        The expected values are formed from the coverage this fixture is built to give
        -- one arm outside the triple-coincident stretch, two inside -- rather than from
        the ``n_trials`` column, so an implementation that wrote a constant factor into
        both columns consistently still fails.
        """
        model = _model()
        table = _candidates(
            ["outside", "inside"], [T0 + 100.0, T0 + 500.0], [2.0, 0.3]
        )
        records = build_records(table, model)
        corrected = apply(table, records, model, observation_time_s=HL_LIVETIME_S)

        expected = np.array([1, 2]) * table["far_per_yr"]
        assert np.array_equal(corrected["far_trials_per_yr"], expected)

    def test_corrected_ifar_is_the_quotient(self):
        """
        ifar_trials_yr equals ifar_yr divided by n_trials, exactly.

        Taken from the stored IFAR rather than re-derived as ``1 / far_trials_per_yr``,
        which also preserves the cap the FAR curve applies. The two differ in the last
        bits for this candidate, and the test asserts that they do, so it cannot pass
        for the re-derived implementation.
        """
        model = _model(arms=("hl", "lv", "hlv"))
        table = _candidates(["c"], [T0 + 500.0], [0.7])
        records = build_records(table, model)
        assert records[0].n_trials == 3

        corrected = apply(table, records, model, observation_time_s=HL_LIVETIME_S)
        assert corrected["ifar_trials_yr"][0] == table["ifar_yr"][0] / 3
        assert corrected["ifar_trials_yr"][0] != 1.0 / (3 * table["far_per_yr"][0])

    def test_pastro_is_not_scaled(self):
        """
        p_astro is left alone by the correction.

        It is a posterior from a rate mixture, not a tail probability, so multiplying it
        by a trials factor is not a defined operation and would produce a number above
        one for a confident candidate.
        """
        model = _model()
        table = _candidates(["c"], [T0 + 500.0], [0.3], p_astro=[0.97])
        records = build_records(table, model)
        corrected = apply(table, records, model, observation_time_s=HL_LIVETIME_S)

        assert corrected["n_trials"][0] == 2
        assert np.array_equal(corrected["p_astro"], table["p_astro"])
        assert corrected["p_astro"][0] <= 1.0
        assert not [key for key in corrected if key.startswith("p_astro_trials")]

    def test_both_tiers_are_written(self):
        """
        tier and tier_trials are both present and independently derived.

        The first candidate is inside the parameter-estimation threshold on its own rate
        and outside it once the factor is applied, so a tier column copied from the
        other fails. The second is comfortably inside in both views, so the test is not
        merely asserting that the two always disagree.
        """
        model = _model(arms=("hl", "lv", "hlv"))
        table = _candidates(
            ["demoted", "safe"],
            [T0 + 500.0, T0 + 500.0 + 0.0],
            [0.4, 0.1],
            p_astro=[0.9, 0.99],
        )
        table["name"] = np.asarray(["demoted", "safe"])
        records = build_records(table, model)
        corrected = apply(table, records, model, observation_time_s=HL_LIVETIME_S)

        assert corrected["tier"].tolist() == [TIER_PE, TIER_PE]
        assert corrected["tier_trials"].tolist() == [TIER_CONFIDENT, TIER_PE]

    def test_without_trials_round_trips(self):
        """
        The uncorrected view recovers the table as it was before the correction.

        Both views must be available from one stored campaign, so the correction is a
        presentation choice rather than a destructive step. Every column of the original
        is compared bit for bit, so a view rebuilt by dividing the corrected numbers
        back out -- which does not round-trip -- would fail.
        """
        model = _model(arms=("hl", "lv", "hlv"))
        table = _candidates(["c"], [T0 + 500.0], [0.7], p_astro=[0.9])
        records = build_records(table, model)
        corrected = apply(table, records, model, observation_time_s=HL_LIVETIME_S)
        recovered = without_trials(corrected)

        for key, values in table.items():
            assert np.array_equal(recovered[key], values), key
        for key in ("far_trials_per_yr", "ifar_trials_yr", "p_value_trials",
                    "tier_trials"):
            assert key not in recovered
        assert recovered["n_trials"].tolist() == [1]
        assert recovered["trials_convention"].tolist() == ["none"]

    def test_comparison_flags_threshold_crossings(self):
        """
        A candidate that qualifies in one view and not the other is reported.

        That set is the point of publishing both views; it is invisible if only the
        corrected table is released. The crossing candidate sits just inside the
        two-per-day inclusion threshold on its own rate and just outside it once
        doubled; the other stays inside in both views, so a comparison that flagged
        every multiply-covered candidate would fail.
        """
        model = _model()
        table = _candidates(
            ["crosses", "stays"],
            [T0 + 500.0, T0 + 550.0],
            [700.0, 1.0],
            p_astro=[0.2, 0.9],
        )
        records = build_records(table, model)
        corrected = apply(table, records, model, observation_time_s=HL_LIVETIME_S)
        report = comparison(corrected, records)

        assert report["crossings"] == ("crosses",)
        assert report["n_crossings"] == 1
        assert report["included"].tolist() == [True, True]
        assert report["included_trials"].tolist() == [False, True]


class TestArmAssignment:
    """Reporting a multiply-found candidate once."""

    def test_best_arm_is_the_most_significant(self):
        """
        A candidate found by several arms is quoted under the one that ranked it
        highest.

        Ranked by IFAR and not by the raw statistic: the arms have different
        backgrounds, so the same statistic is not the same significance in both. Here
        the louder statistic belongs to the arm with the noisier background, so an
        implementation
        comparing statistics picks the other arm and fails -- and ``prefer="stat"`` is
        asserted to pick it, which shows the difference is the ranking and not the
        fixture.
        """
        model = _model()
        table = _candidates(["c"], [T0 + 500.0], [1.0])
        curves = {
            "hl": _curve(np.arange(1.0, 41.0)),
            "hlv": _curve(np.arange(1.0, 11.0) * 0.5),
        }
        records = build_records(
            table,
            model,
            triggers_by_arm={
                "hl": _triggers([T0 + 500.0], [12.0]),
                "hlv": _triggers([T0 + 500.03], [10.0]),
            },
            curves=curves,
        )
        assert records[0].found_stat["hl"] > records[0].found_stat["hlv"]
        assert records[0].found_ifar_yr["hlv"] > records[0].found_ifar_yr["hl"]
        assert assign_best_arm(table, records) == ["hlv"]
        assert assign_best_arm(table, records, prefer="stat") == ["hl"]
        assert records[0].best_arm == "hlv"

    def test_multiply_found_candidate_appears_once(self):
        """The same event seen by two arms yields one row, not two."""
        model = _model()
        table = _candidates(["c"], [T0 + 500.0], [1.0])
        records = build_records(
            table,
            model,
            triggers_by_arm={
                "hl": _triggers([T0 + 499.98], [11.0]),
                "hlv": _triggers([T0 + 500.04], [10.5]),
            },
        )
        assert len(records) == 1
        assert records[0].found_by == ("hl", "hlv")
        assert records[0].is_multiply_found

        corrected = apply(table, records, model, observation_time_s=HL_LIVETIME_S)
        assert corrected["name"].tolist() == ["c"]
        assert corrected["found_by"].tolist() == ["hl,hlv"]
        assert summary(records, model)["n_multiply_found"] == 1

    def test_match_window_exceeds_light_travel(self):
        """
        Two arms' estimates of the same event merge; distinct events do not.

        The window has to exceed the network's light-travel time, since the arms
        estimate arrival time independently and will not agree exactly. The bound is
        taken from
        the detector geometry rather than restated here, so it tracks the network
        actually being searched.
        """
        pytest.importorskip("pycbc")
        light_travel_s = GEOMETRY.max_light_travel_s(("H1", "L1", "V1"))
        assert 0.1 > light_travel_s

        model = _model()
        table = _candidates(["c"], [T0 + 500.0], [1.0])
        merged = build_records(
            table,
            model,
            triggers_by_arm={
                "hl": _triggers([T0 + 500.0], [11.0]),
                "hlv": _triggers([T0 + 500.0 + 0.05], [10.0]),
            },
        )
        assert merged[0].found_by == ("hl", "hlv")

        # Five seconds apart is a different event, not a disagreement about the time.
        separate = build_records(
            table,
            model,
            triggers_by_arm={
                "hl": _triggers([T0 + 500.0], [11.0]),
                "hlv": _triggers([T0 + 505.0], [10.0]),
            },
        )
        assert separate[0].found_by == ("hl",)


class TestModel:
    """The arm registry and its provenance."""

    def test_duplicate_arm_key_is_refused(self):
        """Registering the same arm twice raises rather than replacing."""
        model = _model()
        with pytest.raises(ValueError, match="already registered"):
            model.add(
                SearchArm(key="hl", detectors=("H1", "V1"), observing_run="O3a"),
                ArmSegments(arm="hl", intervals=_intervals((T0, T0 + 10.0))),
            )
        assert model.arms["hl"].detectors == ("H1", "L1")

    def test_arm_requires_distinct_detectors(self):
        """An arm naming no detectors, or one twice, is rejected at construction."""
        with pytest.raises(ValueError, match="no detectors"):
            SearchArm(key="empty", detectors=(), observing_run="O3a")
        with pytest.raises(ValueError, match="repeats a detector"):
            SearchArm(key="hh", detectors=("H1", "H1"), observing_run="O3a")

    def test_model_round_trips(self, tmp_path):
        """
        A saved model reloads with the same arms, segments and convention.

        Interval endpoints are compared exactly, not approximately: they are GPS times
        of order 1.2e9, where a float written at less than full precision moves a
        boundary by more than a window and silently changes which arms covered a
        candidate. Coverage is recomputed from the reloaded model for the same reason.
        """
        model = _model(external=True)
        path = model.save(tmp_path / "trials_model.json")
        reloaded = TrialsModel.load(path)

        assert reloaded.convention == model.convention
        assert list(reloaded.arms) == list(model.arms)
        for key, arm in model.arms.items():
            assert reloaded.arms[key] == arm
            assert np.array_equal(
                reloaded.segments[key].intervals, model.segments[key].intervals
            )
            assert reloaded.segments[key].livetime_s == model.segments[key].livetime_s

        times = np.array([T0 + 100.0, T0 + 500.0])
        assert reloaded.coverage_at(times) == model.coverage_at(times)

    def test_describe_states_the_convention(self):
        """
        The readable description names every arm and the convention used.

        It also has to say that the factor is an upper bound: arms sharing detectors do
        not have independent noise, and a methods section quoting the factor without
        that sentence claims an independence the campaign does not have.
        """
        text = _model(external=True).describe()
        for key in ("hl", "hlv", "gwtc3"):
            assert key in text
        assert "coverage" in text
        assert "excluded from the factor" in text
        assert "upper bound" in text
