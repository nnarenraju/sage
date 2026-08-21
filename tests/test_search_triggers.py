#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_triggers.py
Description   : The shard schema, the trigger table and the histogram algebra.

Created on 2026-08-16

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Histograms are produced by hundreds of separate jobs and added together, so addition has
to be exact and has to refuse the combinations that are not meaningful. A raw network
output is unbounded, which makes the entries outside the grid the loudest triggers in the
campaign rather than a rounding detail.

Runs anywhere; needs no data, no GPU and no network.
"""

import numpy as np
import pytest

from sage.search.triggers import (
    STAT_HIST_HI,
    STAT_HIST_LO,
    STAT_HIST_NBINS,
    StatHistogram,
    TriggerTable,
    hist_edges,
    histogram_stats,
)


def _table(n=4, **overrides):
    """A small schema-valid trigger table."""
    columns = {
        "stat": np.linspace(5.0, 9.0, n),
        "gps": 1238166018.0 + np.arange(n) * 10.0,
        "slide_id": np.zeros(n, dtype=np.int64),
    }
    columns.update(overrides.pop("columns", {}))
    attrs = {"spec_hash": "abc", "observing_run": "O3a", "arm": "HL", "clustered": True}
    attrs.update(overrides.pop("attrs", {}))
    return TriggerTable(columns=columns, attrs=attrs)


class TestGrid:
    """The shared binning."""

    def test_edges_span_the_declared_range(self):
        """The grid is the declared constants, so every shard agrees on it."""
        edges = hist_edges()
        assert edges.size == STAT_HIST_NBINS + 1
        assert edges[0] == STAT_HIST_LO
        assert edges[-1] == STAT_HIST_HI

    def test_edges_are_reproducible(self):
        """Two calls give identical edges, bit for bit."""
        assert np.array_equal(hist_edges(), hist_edges())


class TestHistogramExactness:
    """Addition over shards is the operation the background depends on."""

    def test_addition_matches_binning_the_union(self):
        """
        Adding two shards' histograms equals binning their triggers together.

        Background is accumulated by summing histograms from hundreds of jobs, so this is
        the identity the total count rests on. Asserted with integer equality, not
        approximately: one event either way changes every FAR taken from the count.
        """
        rng = np.random.default_rng(1)
        left = rng.normal(loc=5.0, scale=8.0, size=5000)
        right = rng.normal(loc=-2.0, scale=12.0, size=3000)

        combined = histogram_stats(left, clustered=True) + histogram_stats(
            right, clustered=True
        )
        direct = histogram_stats(np.concatenate([left, right]), clustered=True)

        assert np.array_equal(combined.counts, direct.counts)
        assert combined.underflow == direct.underflow
        assert combined.overflow == direct.overflow

    def test_outside_grid_is_kept_not_clipped(self):
        """
        Values beyond the grid are counted separately, not folded into the end bins.

        A raw network output is unbounded, so the entries above the top edge are the
        loudest triggers in the campaign -- the ones a FAR is actually asked about.
        Clipping them into the top bin makes them indistinguishable from merely loud
        ones; dropping them removes them from the background altogether.
        """
        stat = np.array([-1e3, -50.0, 0.0, 50.0, 1e3, 1e30])
        hist = histogram_stats(stat, clustered=True)

        assert hist.underflow == 2      # -1e3 and -50 are below -40
        assert hist.overflow == 2       # 1e3 and 1e30 are above 60
        assert int(hist.counts.sum()) == 2
        assert hist.total == stat.size

    def test_total_counts_every_entry(self):
        """Nothing is lost between the grid and the two outside counters."""
        rng = np.random.default_rng(2)
        stat = rng.normal(loc=0.0, scale=40.0, size=20000)
        assert histogram_stats(stat, clustered=False).total == stat.size

    def test_counts_are_integers(self):
        """
        A float histogram is refused.

        Counts are summed across a whole campaign; in floating point that stops being
        exact past 2**53, and exactness is the only reason to store counts rather than
        the statistics themselves.
        """
        with pytest.raises(TypeError, match="integer dtype"):
            StatHistogram(
                counts=np.zeros(STAT_HIST_NBINS, dtype=np.float64),
                underflow=0,
                overflow=0,
                clustered=True,
            )

    def test_wrong_grid_refused(self):
        """A histogram on a different grid cannot be added and is refused on sight."""
        with pytest.raises(ValueError, match="bins against the shared grid"):
            StatHistogram(
                counts=np.zeros(128, dtype=np.int64),
                underflow=0,
                overflow=0,
                clustered=True,
            )

    def test_nan_refused(self):
        """
        A NaN statistic is a fault to report, not a value to bin.

        It falls in no bin and in neither outside counter, so it would leave the
        histogram silently short of the triggers it claims to describe.
        """
        with pytest.raises(ValueError, match="NaN"):
            histogram_stats(np.array([1.0, np.nan, 2.0]), clustered=True)

    def test_infinities_land_in_overflow(self):
        """+/-inf are extreme values, not faults, and are counted at the ends."""
        hist = histogram_stats(np.array([np.inf, -np.inf]), clustered=True)
        assert hist.overflow == 1
        assert hist.underflow == 1
        assert hist.total == 2


class TestClusteredFlag:
    """The flag that prevents a background from being counted unclustered."""

    def test_mixed_clustering_refused(self):
        """
        A clustered histogram cannot be added to an unclustered one.

        The sum is neither, and afterwards nothing distinguishes it from a valid
        histogram. Counting an unclustered background inflates the event count several
        times over -- the failure that invalidated the reference analysis -- so the two
        are kept apart by the type rather than by a convention.
        """
        clustered = histogram_stats(np.array([1.0]), clustered=True)
        raw = histogram_stats(np.array([1.0]), clustered=False)
        with pytest.raises(ValueError, match="clustered"):
            _ = clustered + raw

    def test_flag_survives_addition(self):
        """The sum of two clustered histograms is still marked clustered."""
        a = histogram_stats(np.array([1.0]), clustered=True)
        b = histogram_stats(np.array([2.0]), clustered=True)
        assert (a + b).clustered is True
        c = histogram_stats(np.array([1.0]), clustered=False)
        assert (c + c).clustered is False


class TestCounting:
    """What a FAR asks a histogram."""

    def test_n_above_is_inclusive(self):
        """
        A background event exactly as loud as the candidate counts toward it.

        It is evidence that the noise reaches that value, so excluding it would make the
        rate too small at exactly the statistic being asked about.
        """
        edges = hist_edges()
        hist = histogram_stats(np.array([edges[100], edges[200], edges[300]]), True)
        assert hist.n_above(edges[200]) == 2
        assert hist.n_above(edges[300]) == 1
        assert hist.n_above(edges[301]) == 0

    def test_n_above_counts_overflow(self):
        """
        Overflow entries are above every query on the grid.

        They are the loudest triggers there are; omitting them from the count would give
        the loudest candidates a rate of zero louder events when several exist.
        """
        hist = histogram_stats(np.array([0.0, 1e6, 1e6]), clustered=True)
        assert hist.n_above(50.0) == 2
        assert hist.n_above(STAT_HIST_HI + 1.0) == 2

    def test_n_above_below_grid_counts_everything(self):
        """A query under the whole grid returns every entry, underflow included."""
        hist = histogram_stats(np.array([-1e3, 0.0, 1e3]), clustered=True)
        assert hist.n_above(STAT_HIST_LO - 1.0) == 3

    def test_n_above_is_monotone(self):
        """The count never rises as the query does."""
        rng = np.random.default_rng(4)
        hist = histogram_stats(rng.normal(scale=20.0, size=5000), clustered=True)
        queries = np.linspace(STAT_HIST_LO - 5.0, STAT_HIST_HI + 5.0, 400)
        counts = [hist.n_above(q) for q in queries]
        assert all(a >= b for a, b in zip(counts, counts[1:]))

    def test_quantile_keeps_at_least_the_rate(self):
        """
        The threshold retains at least the fraction asked for, never less.

        It freezes the campaign's keep threshold from the complete zero-lag histogram.
        Falling short would discard triggers the configuration said to keep, and the
        shortfall would only surface as a background thinner than it should be.
        """
        rng = np.random.default_rng(5)
        stat = rng.normal(loc=0.0, scale=10.0, size=100000)
        hist = histogram_stats(stat, clustered=False)
        for rate in (0.5, 0.1, 0.01, 1e-3):
            threshold = hist.quantile_threshold(rate)
            assert hist.n_above(threshold) >= rate * hist.total

    def test_quantile_returns_a_bin_edge(self):
        """
        The threshold is exactly representable, so every job uses the identical number.

        It is written once and read by hundreds of slide jobs; a value that was not a
        grid point could be rounded differently on the way in and out, and the jobs would
        threshold on numbers that differ in the last bit.
        """
        rng = np.random.default_rng(6)
        hist = histogram_stats(rng.normal(scale=10.0, size=10000), clustered=False)
        assert float(hist.quantile_threshold(0.05)) in set(hist_edges().tolist())

    def test_quantile_rejects_bad_rate(self):
        """A rate outside (0, 1] is a configuration error, not a clamp."""
        hist = histogram_stats(np.array([1.0]), clustered=False)
        for rate in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="keep_rate"):
                hist.quantile_threshold(rate)


class TestTriggerTable:
    """The in-memory trigger set."""

    def test_length_and_column_access(self):
        table = _table(n=4)
        assert len(table) == 4
        assert table["stat"].shape == (4,)
        with pytest.raises(KeyError, match="mchirp"):
            _ = table["mchirp"]

    def test_unknown_column_refused(self):
        """
        A column outside the schema is refused at construction.

        Shards are read by stages that know only the declared columns, so an extra one is
        written and then dropped by the first stage that copies the table -- losing it
        without a word at whichever stage happens to copy first.
        """
        with pytest.raises(ValueError, match="not in the shard schema"):
            TriggerTable(columns={"snr": np.zeros(3)}, attrs={})

    def test_unequal_columns_refused(self):
        """Columns of different lengths do not describe one set of triggers."""
        with pytest.raises(ValueError, match="unequal length"):
            TriggerTable(
                columns={"stat": np.zeros(3), "gps": np.zeros(4)}, attrs={}
            )

    def test_sort_is_stable(self):
        """
        Equal sort keys keep production order.

        Clustering breaks ties toward the earlier trigger, so an unstable sort would make
        which trigger that is depend on the sort implementation rather than on the data.
        """
        table = TriggerTable(
            columns={
                "gps": np.array([1.0, 1.0, 1.0, 0.0]),
                "stat": np.array([10.0, 20.0, 30.0, 40.0]),
            },
            attrs={},
        )
        assert table.sort_by("gps")["stat"].tolist() == [40.0, 10.0, 20.0, 30.0]

    def test_filter_requires_boolean_mask(self):
        """
        An integer mask is refused rather than read as fancy indexing.

        ``table.filter(np.array([0, 1]))`` would silently return the first two triggers
        rather than the two the caller selected, and both results are plausible tables.
        """
        table = _table(n=4)
        with pytest.raises(TypeError, match="boolean"):
            table.filter(np.array([0, 1]))

    def test_filter_selects_subset(self):
        table = _table(n=4)
        kept = table.filter(table["stat"] > 6.0)
        assert len(kept) < 4
        assert np.all(kept["stat"] > 6.0)

    def test_concat_joins_columns(self):
        table = _table(n=3)
        joined = table.concat(_table(n=2))
        assert len(joined) == 5
        assert joined.attrs["observing_run"] == "O3a"

    @pytest.mark.parametrize(
        "key,value",
        [
            ("spec_hash", "different"),
            ("observing_run", "O3b"),
            ("arm", "HLV"),
            ("clustered", False),
        ],
    )
    def test_concat_refuses_incompatible_provenance(self, key, value):
        """
        Shards from different analyses are not one trigger set.

        Each of these changes what a trigger means -- which configuration produced it,
        which run and network it belongs to, whether it has been clustered. Joining
        across any of them builds a background out of analyses that were never
        comparable, and the joined table looks exactly like a valid one.
        """
        with pytest.raises(ValueError, match=key):
            _table(n=2).concat(_table(n=2, attrs={key: value}))

    def test_concat_drops_disagreeing_attrs(self):
        """
        An attribute the two shards disagree on is dropped, not silently taken from one.

        Keeping the left-hand shard's slide id on a table holding both would label every
        trigger with a slide most of them did not come from.
        """
        joined = _table(n=2, attrs={"slide_id": 1}).concat(
            _table(n=2, attrs={"slide_id": 2})
        )
        assert "slide_id" not in joined.attrs
        assert joined.attrs["arm"] == "HL"
