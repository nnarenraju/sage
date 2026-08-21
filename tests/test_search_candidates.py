#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_candidates.py
Description   : The candidate table, its tiers, and what it refuses to publish.

Created on 2026-08-20

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later

This table is the single source of truth for every downstream figure, table and release
artefact, so its failures are the ones that reach a paper. Three kinds are tested here:
joins that pair the wrong rows, tiers that claim more than was decided, and columns that
would present a number as a measurement when it is not.
"""

import numpy as np
import pytest

from sage.search.candidates import (
    TIER_CANDIDATE,
    TIER_CONFIDENT,
    TIER_PE,
    TIER_UNDETERMINED,
    CandidateTable,
    apply_tiers,
    expected_contamination,
    from_triggers,
    recompute_far,
)

pytest.importorskip("h5py")


def _curve(seed=2, livetime_yr=50.0, foreground_yr=0.3):
    """A FAR curve over a background long enough to resolve a per-year rate."""
    from sage.search.background import BackgroundSet
    from sage.search.far import build_far_curve
    from sage.search.triggers import histogram_stats

    stats = np.random.default_rng(seed).exponential(1.0, 50_000) + 1.0
    background = BackgroundSet(
        stats=stats,
        livetime_s=3.15576e7 * livetime_yr,
        n_slides=8,
        removal="inclusive",
        histogram=histogram_stats(stats, clustered=True),
    )
    return build_far_curve(
        background,
        foreground_livetime_s=3.15576e7 * foreground_yr,
        tail=None,
        ifar_cap_yr=1000.0,
    )


def _clustered(stats, gps=None):
    """A clustered zero-lag trigger table."""
    from sage.search.triggers import TriggerTable

    stats = np.asarray(stats, dtype=np.float64)
    if gps is None:
        gps = 1.2381e9 + np.arange(stats.size) * 100.0
    return TriggerTable(
        columns={
            "stat": stats,
            "gps": np.asarray(gps, dtype=np.float64),
            "tc_gps": np.asarray(gps, dtype=np.float64),
            "mchirp": np.full(stats.size, 25.0),
        },
        attrs={"clustered": True},
    )


class TestAssembly:
    """What goes into the table, and where each number comes from."""

    def test_unclustered_is_refused(self):
        """
        One event spans many analysis windows. An unclustered list would enter the same
        event once per window, each copy with its own name and rate.
        """
        table = _clustered([10.0, 8.0])
        table.attrs["clustered"] = False
        with pytest.raises(ValueError, match="clustered"):
            from_triggers(table, _curve())

    def test_rates_come_from_the_curve(self):
        """
        A candidate's rate is the number the FAR stage published, read from the curve
        rather than recomputed, so the two cannot drift apart.
        """
        curve = _curve()
        stats = np.array([12.0, 9.0, 6.0])
        table = from_triggers(_clustered(stats), curve)
        assert np.allclose(
            table.columns["far_per_yr"], curve.far_of(np.sort(stats)[::-1])
        )

    def test_sorted_loudest_first(self):
        """The list is ordered by significance, which is how it is read."""
        table = from_triggers(_clustered([6.0, 12.0, 9.0]), _curve())
        assert np.all(np.diff(table.columns["stat"]) <= 0)

    def test_names_are_unique(self):
        """
        The name is the identity every later join uses. Clustering separates candidates
        by 0.35 s, so several share a second and second-resolution names collide.
        """
        gps = 1.2381e9 + np.array([0.0, 0.4, 0.8, 1.2])
        table = from_triggers(_clustered([9.0, 8.0, 7.0, 6.0], gps=gps), _curve())
        names = [str(name) for name in table.columns["name"]]
        assert len(set(names)) == len(names)

    def test_tiers_start_undetermined(self):
        """Assembly does not decide tiers; apply_tiers does."""
        table = from_triggers(_clustered([12.0, 9.0]), _curve())
        assert np.all(table.columns["tier"] == TIER_UNDETERMINED)


class TestTiers:
    """The ladder, and what it refuses to claim."""

    def _table(self, p_astro):
        table = from_triggers(_clustered([12.0, 9.0, 6.0]), _curve())
        table.columns["p_astro"] = np.asarray(p_astro, dtype=np.float64)
        return table

    def test_confident_needs_a_probability(self):
        """
        Without p_astro nothing is promoted past the broad list. The confident tier is
        defined by a probability, and treating its absence as passing would promote every
        loud candidate on rate alone.
        """
        table = from_triggers(_clustered([12.0, 9.0]), _curve())
        tiered = apply_tiers(table, far_candidate_per_day=1e6)
        assert np.all(tiered.columns["tier"] == TIER_CANDIDATE)

    def test_nan_probability_does_not_promote(self):
        """
        A candidate outside the p_astro support carries nan. NaN loses every comparison,
        which is the intended reading: no probability, no promotion.
        """
        tiered = apply_tiers(self._table([np.nan, 0.99, 0.99]), far_candidate_per_day=1e6)
        assert tiered.columns["tier"][0] == TIER_CANDIDATE

    def test_thresholds_are_strict(self):
        """A candidate exactly on a boundary is excluded, not admitted."""
        tiered = apply_tiers(
            self._table([0.5, 0.5, 0.5]),
            far_candidate_per_day=1e6,
            p_astro_confident=0.5,
        )
        assert np.all(tiered.columns["tier"] == TIER_CANDIDATE)

    def test_below_threshold_is_dropped_not_undetermined(self):
        """
        The ladder starts at "in the public list" and has no rung below it. Reusing
        the undetermined marker would say "not yet decided" about a decided row.
        """
        tiered = apply_tiers(self._table([0.9, 0.9, 0.9]), far_candidate_per_day=1e-9)
        assert len(tiered) == 0
        assert tiered.attrs["n_below_threshold"] == 3

    def test_dataquality_requires_a_verdict(self):
        """
        Asking for vetted tiers without a verdict would publish a vetted tier that was
        never vetted.
        """
        with pytest.raises(ValueError, match="dq_vetoed"):
            apply_tiers(
                self._table([0.9, 0.9, 0.9]),
                far_candidate_per_day=1e6,
                use_dataquality=True,
            )


class TestPersistence:
    """What the table refuses to write, and what it refuses to read back."""

    def test_undetermined_tiers_refused_on_load(self, tmp_path):
        """
        Publishing a list whose tiers were never decided is the failure the default
        guards against.
        """
        table = from_triggers(_clustered([12.0, 9.0]), _curve())
        path = tmp_path / "candidates.h5"
        table.save(path)
        with pytest.raises(ValueError, match="undetermined"):
            CandidateTable.load(path)
        assert len(CandidateTable.load(path, allow_undetermined=True)) == 2

    def test_round_trip_is_exact(self, tmp_path):
        """Times of order 1.2e9 must return bit-identical, or a join moves."""
        table = apply_tiers(
            from_triggers(_clustered([12.0, 9.0]), _curve()), far_candidate_per_day=1e6
        )
        path = tmp_path / "candidates.h5"
        table.save(path)
        back = CandidateTable.load(path)
        assert np.array_equal(back.columns["gps"], table.columns["gps"])
        assert [str(v) for v in back.columns["name"]] == [
            str(v) for v in table.columns["name"]
        ]

    def test_fabricated_masses_refused(self, tmp_path):
        """
        Sage estimates tc and mchirp. A mass column can only have come from parameter
        estimation, and a follow-up template's parameters are not a measurement.
        """
        table = apply_tiers(
            from_triggers(_clustered([12.0, 9.0]), _curve()), far_candidate_per_day=1e6
        )
        table.columns["mass1"] = np.array([35.0, 30.0])
        with pytest.raises(ValueError, match="mass_provenance"):
            table.save(tmp_path / "candidates.h5")

    def test_tier_query_refuses_undetermined(self):
        """A tier query cannot be answered for a row that was never decided."""
        table = from_triggers(_clustered([12.0, 9.0]), _curve())
        with pytest.raises(ValueError, match="undetermined"):
            table.tier(TIER_CONFIDENT)


class TestDerived:
    """Quantities read off the table."""

    def test_contamination_is_summed_per_tier(self):
        """
        "N confident candidates, of which an expected M are terrestrial" is a statement
        about the confident subset; the broad list's number does not bound it.
        """
        table = from_triggers(_clustered([12.0, 9.0, 6.0]), _curve())
        table.columns["p_astro"] = np.array([0.99, 0.6, 0.2])
        tiered = apply_tiers(table, far_candidate_per_day=1e6)
        report = expected_contamination(tiered)
        assert report["n_candidates"] == 3
        assert np.isclose(report["expected_terrestrial"], 0.01 + 0.4 + 0.8)
        assert report["expected_terrestrial_confident"] <= report["expected_terrestrial"]

    def test_recompute_far_resets_tiers(self):
        """
        Tiers derived from the old rates beside new rates is an inconsistency nothing
        later would detect, since tier() would answer from them without complaint.
        """
        table = apply_tiers(
            from_triggers(_clustered([12.0, 9.0]), _curve()), far_candidate_per_day=1e6
        )
        again = recompute_far(table, _curve(livetime_yr=100.0))
        assert np.all(again.columns["tier"] == TIER_UNDETERMINED)
        assert again.attrs["provisional_tiers"]
