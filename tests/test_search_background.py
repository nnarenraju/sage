#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_background.py
Description   : Slide collation, the three removal modes and the dispersion test.

Created on 2026-08-16

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The background event count is the numerator of every false-alarm rate the search
reports, so the failures worth testing here are the ones that change that count while
leaving every product looking ordinary: clustering the slides together instead of
separately, taking the livetime from a closed form instead of from the measured ladder,
or removing foreground in an order that depends on how the table was written.

Fixtures are built so the answer is known by construction rather than by re-running
the implementation, and the sandwich between the three backgrounds is asserted across
a range of statistics rather than at one point -- a single point is passed by an
implementation that returns the inclusive set unchanged.

Runs anywhere; needs no data, no GPU and no network.
"""

from pathlib import Path

import numpy as np
import pytest

from sage.search import background as bg_module
from sage.search import far as far_module
from sage.search.background import (
    SECONDS_PER_JULIAN_YEAR,
    BackgroundSet,
    collate_slides,
    exclusive_background,
    far_of_stat,
    hierarchical_removal,
    n_louder,
    overdispersion_lrt,
)
from sage.search.geometry import SearchGeometry
from sage.search.segments import Segment
from sage.search.manifest import SCHEMA_VERSION
from sage.search.slides import Slide, SlidePlan
from sage.search.triggers import TriggerTable, TriggerWriter, histogram_stats

T0 = 1238166018.0

# Long enough that a candidate louder than the whole background clears a per-year
# threshold: (1 + 0) / T is 0.316 per year here, so the removal decisions below turn on
# the count rather than on the length of the ladder.
LIVETIME_S = 1.0e8

# Per-slide livetimes of the hand-built plan. Deliberately unequal and all below the
# zero-lag slide's, which is what retention falling with lag looks like, so the summed
# background time cannot be reproduced by any multiple of the foreground time.
ZEROLAG_LIVETIME_S = 1000.0
SLID_LIVETIMES_S = (900.0, 800.0)

_PROVENANCE = {
    "schema_version": SCHEMA_VERSION,
    "sage_version": "0.0.1",
    "git_hash": "0" * 40,
    "git_dirty": False,
    "spec_hash": "deadbeef",
    "config_module": "tests.test_search_background",
    "checkpoint_path": "",
    "checkpoint_sha256": "",
    "observing_run": "O3a",
    # O3b-trained, O3a-searched: the background is drawn from data the network never saw.
    "train_runs": ("O3b",),
    "detectors": ("H1", "L1"),
    "sample_rate": 2048.0,
    "window_samples": 24576,
    "stride_samples": 205,
    "seed": 0,
    "created_utc": "2026-08-16T00:00:00Z",
    "arm": "HL",
    "clustered": False,
}


def _write_shard(path, gps, stat, slide_id=None):
    """
    Write one unclustered slide shard through the production writer.

    Written with :class:`~sage.search.triggers.TriggerWriter` rather than assembled by
    hand, so the collation is exercised against the layout the search actually
    produces -- committed row counts, stamped provenance and all -- instead of against
    a file shaped to suit the test.
    """
    stat = np.asarray(stat, dtype=np.float64)
    columns = {"gps": np.asarray(gps, dtype=np.float64), "stat": stat}
    if slide_id is not None:
        columns["slide_id"] = np.asarray(slide_id, dtype=np.int64)
    writer = TriggerWriter(path, dict(_PROVENANCE))
    writer.append(
        TriggerTable(
            columns=columns,
            attrs={
                "spec_hash": _PROVENANCE["spec_hash"],
                "observing_run": _PROVENANCE["observing_run"],
                "arm": _PROVENANCE["arm"],
                "clustered": False,
            },
        )
    )
    writer.add_histogram(histogram_stats(stat, clustered=False))
    writer.complete_block(0)
    writer.close()
    return Path(path)


def _plan():
    """
    A slide plan with measured, unequal per-slide livetimes.

    Constructed directly rather than through :meth:`SlidePlan.build`, because what the
    collation must be pinned against is the arithmetic on the livetimes, and a drawn
    ladder makes the expected total depend on the lag draw.
    """
    slides = [Slide(0, {"H1": 0.0, "L1": 0.0}, 100, ZEROLAG_LIVETIME_S)]
    for index, livetime in enumerate(SLID_LIVETIMES_S):
        slides.append(
            Slide(index + 1, {"H1": 0.0, "L1": 50.0 * (index + 1)}, 90, livetime)
        )
    return SlidePlan(
        slides=slides,
        reference_detector="H1",
        seed=0,
        min_separation_s=20.0,
        tau_max_s=8192.0,
    )


def _inclusive():
    """
    A background whose loud tail sits entirely inside one veto window.

    Four hundred ordinary events spread over the run, all below stat 8, plus three loud
    ones packed within a second of ``T0 + 500``. That is what makes the removal of the
    candidate at ``T0 + 500`` change the count seen by the quieter candidate at
    ``T0 + 9000``: without the hot spot the two candidates would be independent and
    the hierarchical set would equal the inclusive one at every statistic worth asking
    about.

    The ordinary events are offset off the twenty-second lattice so that none of them
    falls inside the loud candidate's veto window, which keeps the number of events that
    window removes a property of the fixture rather than of the offset.
    """
    rng = np.random.default_rng(7)
    times = T0 + 5.0 + np.arange(400) * 20.0
    stats = 2.0 + 6.0 * rng.random(400)
    times = np.concatenate([times, [T0 + 499.5, T0 + 500.5, T0 + 501.0]])
    stats = np.concatenate([stats, [12.0, 11.0, 10.5]])
    order = np.argsort(times)
    times, stats = times[order], stats[order]
    return BackgroundSet(
        stats=stats,
        livetime_s=LIVETIME_S,
        n_slides=8,
        removal="inclusive",
        histogram=histogram_stats(stats, clustered=True),
        gps=times,
    )


def _zerolag(stat, gps, clustered=True):
    """A clustered zero-lag candidate list in the schema the removal stages read."""
    return TriggerTable(
        columns={
            "stat": np.asarray(stat, dtype=np.float64),
            "gps": np.asarray(gps, dtype=np.float64),
        },
        attrs={"clustered": clustered},
    )


def _far_per_yr(stat, stats):
    """Rate in the units the removal threshold is quoted in."""
    return far_of_stat(np.array([stat]), stats, LIVETIME_S)[0] * SECONDS_PER_JULIAN_YEAR


# A real geometry, segment set and ladder, so the removal stages re-measure livetime
# through the same lattice the background is scored on. A one-second stride rather than
# the production 205 samples: nothing asserted about removal depends on the stride, and
# the ladder is rebuilt once per removal in these tests.
REAL_GEOMETRY = SearchGeometry(
    sample_rate=2048.0,
    signal_length_s=12.0,
    padding_length_s=2.0,
    stride_samples=2048,
    tc_lower_s=5.5,
    tc_upper_s=6.5,
)
REAL_CHUNK_S = 4096.0
LOUD_GPS = T0 + 2000.0


def _real_segments(detector, n_chunks=3):
    """Three long chunks with a gap between them, identical in both detectors."""
    rate = 2048.0
    nsamples = int(REAL_CHUNK_S * rate)
    return [
        Segment(
            segment_index=k,
            detector=detector,
            observing_run="O3a",
            gps_start=T0 + k * (REAL_CHUNK_S + 100.0),
            gps_end=T0 + k * (REAL_CHUNK_S + 100.0) + REAL_CHUNK_S,
            sample_rate=rate,
            nsamples=nsamples,
            sample_start_idx=k * nsamples,
            dyn_range_fac=1.0,
            noise_low_freq_cutoff=15.0,
        )
        for k in range(n_chunks)
    ]


def _real_network():
    return {"H1": _real_segments("H1"), "L1": _real_segments("L1")}


def _real_plan(n_slides=4):
    return SlidePlan.build(
        REAL_GEOMETRY,
        _real_network(),
        n_slides=n_slides,
        reference_detector="H1",
        min_separation_s=20.0,
        tau_max_s=1000.0,
        seed=5,
    )


def _real_inclusive(plan):
    """
    A background on the real ladder, carrying two families of contamination.

    The **reference family** sits at ``LOUD_GPS`` in three slides -- what a real Hanford
    signal does to a background, one copy per slide at the same H1 time. The **follower
    family** sits at ``LOUD_GPS - offset_L1`` in three slides, so every one of them lands
    on ``LOUD_GPS`` in L1's own frame: what a real Livingston signal does, at a
    *different* reference time in every slide.

    Both are here deliberately. A fixture carrying only the reference family cannot tell a
    per-detector veto from a reference-frame-only one, which is how a veto that caught
    half the contamination in HL passed its tests.
    """
    slid = [s.slide_id for s in plan if s.slide_id != 0]
    offsets = {int(s.slide_id): dict(s.offsets_s) for s in plan}
    rng = np.random.default_rng(3)
    times, stats, ids = [], [], []
    for slide_id in slid:
        n = 120
        times.append(T0 + 20.0 + np.arange(n) * 31.0)
        stats.append(2.0 + 5.0 * rng.random(n))
        ids.append(np.full(n, slide_id))
    for rank, slide_id in enumerate(slid[:3]):
        times.append(np.array([LOUD_GPS]))
        stats.append(np.array([12.0 - 0.5 * rank]))
        ids.append(np.array([slide_id]))
    for rank, slide_id in enumerate(slid[:3]):
        shift = float(offsets[int(slide_id)].get("L1", 0.0))
        times.append(np.array([LOUD_GPS - shift]))
        stats.append(np.array([11.9 - 0.5 * rank]))
        ids.append(np.array([slide_id]))
    times = np.concatenate(times)
    stats = np.concatenate(stats)
    ids = np.concatenate(ids)
    order = np.argsort(times)
    return BackgroundSet(
        stats=stats[order],
        livetime_s=plan.background_livetime_s,
        n_slides=len(slid),
        removal="inclusive",
        histogram=histogram_stats(stats[order], clustered=True),
        gps=times[order],
        tc_gps=times[order],
        slide_id=ids[order],
    )


def _real_zerolag(stat, gps, clustered=True):
    """A clustered zero-lag list carrying the decoded merger time."""
    gps = np.asarray(gps, dtype=np.float64)
    return TriggerTable(
        columns={
            "stat": np.asarray(stat, dtype=np.float64),
            "gps": gps,
            "tc_gps": gps,
        },
        attrs={"clustered": clustered},
    )


class TestCounting:
    """The count and the rate the whole search divides by."""

    def test_far_reexported_not_reimplemented(self):
        """
        ``far.far_of_stat`` is this module's function, not a second copy of it.

        Two implementations of conservative counting drift apart at exactly the boundary
        cases -- the ``1 +``, the inclusive tie -- and the divergence surfaces as two
        different FARs quoted for one candidate by two stages of the same search.
        """
        assert far_module.far_of_stat is bg_module.far_of_stat
        assert far_module.n_louder is bg_module.n_louder

    def test_count_is_inclusive_at_stat(self):
        """
        A background event exactly as loud as the query counts toward it.

        It is evidence the noise reaches that value, so excluding it understates the
        rate at precisely the statistic being asked about.
        """
        stats = np.array([1.0, 3.0, 3.0, 3.0, 5.0])
        assert n_louder(np.array([3.0]), stats).tolist() == [4]
        assert n_louder(np.array([5.000001]), stats).tolist() == [0]

    def test_count_matches_brute_force(self):
        """
        The sorted-array count agrees with direct comparison at every query.

        The implementation is a ``searchsorted`` on a sorted copy; a wrong ``side``
        gives the right answer everywhere except at values present in the background,
        which is where ties live.
        """
        rng = np.random.default_rng(3)
        stats = np.round(rng.normal(size=500), 2)
        queries = np.concatenate([stats[:50], rng.normal(size=50)])
        direct = [int(np.count_nonzero(stats >= q)) for q in queries]
        assert n_louder(queries, stats).tolist() == direct

    def test_rate_adds_one_to_the_count(self):
        """The conservative numerator, so the loudest candidate gets a finite IFAR."""
        stats = np.array([1.0, 2.0, 3.0])
        assert far_of_stat(np.array([1e6]), stats, 100.0)[0] == pytest.approx(0.01)
        assert far_of_stat(np.array([2.0]), stats, 100.0)[0] == pytest.approx(0.03)

    def test_nan_background_refused(self):
        """
        A NaN background event is refused rather than counted as quiet.

        Every comparison against NaN is false, so it silently drops out of the numerator
        and lowers the rate of every candidate assessed against it.
        """
        with pytest.raises(ValueError, match="NaN"):
            far_of_stat(np.array([1.0]), np.array([1.0, np.nan]), 100.0)

    @pytest.mark.parametrize("livetime", [0.0, -1.0, np.inf, np.nan])
    def test_bad_livetime_refused(self, livetime):
        """A denominator that is not a positive time is a configuration error."""
        with pytest.raises(ValueError, match="livetime"):
            far_of_stat(np.array([1.0]), np.array([1.0]), livetime)


class TestBackgroundSet:
    """What a set has to carry before anything may be counted from it."""

    def test_unclustered_histogram_refused(self):
        """
        An unclustered histogram cannot stand for a background.

        A glitch contributes one entry per window rather than one per event, inflating
        the count several times over -- and since the count is the FAR numerator, every
        rate taken from it is wrong by that factor while looking entirely ordinary.
        """
        with pytest.raises(ValueError, match="clustered"):
            BackgroundSet(
                stats=np.array([1.0]),
                livetime_s=100.0,
                n_slides=1,
                removal="inclusive",
                histogram=histogram_stats(np.array([1.0]), clustered=False),
            )

    def test_unknown_removal_refused(self):
        """
        The removal mode names which of three different backgrounds this is.

        A set labelled with anything else cannot be attributed to a procedure, and the
        three give materially different significances for the same candidate.
        """
        with pytest.raises(ValueError, match="removal must be one of"):
            BackgroundSet(
                stats=np.array([1.0]),
                livetime_s=100.0,
                n_slides=1,
                removal="partial",
            )

    @pytest.mark.parametrize("livetime", [0.0, -5.0, np.nan])
    def test_bad_livetime_refused(self, livetime):
        """Every rate divides by the livetime, so it is checked at construction."""
        with pytest.raises(ValueError, match="livetime"):
            BackgroundSet(
                stats=np.array([1.0]),
                livetime_s=livetime,
                n_slides=1,
                removal="inclusive",
            )

    def test_n_above_prefers_exact_stats(self):
        """
        With statistics present the count is exact, not resolved to a bin.

        The histogram over-counts by up to one bin, which is the right direction but the
        wrong number for reproducing a published IFAR.
        """
        stats = np.array([1.0, 2.0, 2.0, 7.5])
        background = BackgroundSet(
            stats=stats,
            livetime_s=100.0,
            n_slides=1,
            removal="inclusive",
            histogram=histogram_stats(stats, clustered=True),
        )
        assert background.n_above(2.0) == 3
        assert background.n_above(7.5) == 1

    def test_n_above_falls_back_to_histogram(self):
        """
        A set restored from a histogram alone can still be counted.

        That is the whole reason the histogram travels with the statistics: a background
        summarised for a figure remains usable as a rate denominator's numerator.
        """
        stats = np.array([1.0, 2.0, 30.0])
        background = BackgroundSet(
            stats=np.empty(0),
            livetime_s=100.0,
            n_slides=1,
            removal="inclusive",
            histogram=histogram_stats(stats, clustered=True),
        )
        assert background.n_above(30.0) == 1
        assert background.n_above(-1e3) == 3

    def test_empty_set_cannot_be_counted(self):
        """Neither statistics nor a histogram means there is nothing to count."""
        background = BackgroundSet(
            stats=np.empty(0), livetime_s=100.0, n_slides=1, removal="inclusive"
        )
        with pytest.raises(ValueError, match="neither statistics nor a histogram"):
            background.n_above(1.0)


class TestPersistence:
    """The background as it is handed to the next stage and to a reviewer."""

    def test_round_trip_preserves_every_field(self, tmp_path):
        """
        A reloaded set counts identically to the one that was written.

        The statistics are stored rather than only the histogram because they are what
        reproduces the exact count a published IFAR was quoted from; a round trip that
        kept the summary alone would still look complete.
        """
        stats = np.array([1.0, 2.5, 2.5, 9.0, 41.0])
        times = T0 + np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        original = BackgroundSet(
            stats=stats,
            livetime_s=1234.5,
            n_slides=17,
            removal="hierarchical",
            histogram=histogram_stats(stats, clustered=True),
            removed_gps=np.array([T0 + 5.0, T0 + 25.0]),
            gps=times,
        )
        path = tmp_path / "background" / "bg_hierarchical.h5"
        original.save(path)
        restored = BackgroundSet.load(path)

        assert np.array_equal(restored.stats, stats)
        assert np.array_equal(restored.gps, times)
        assert np.array_equal(restored.removed_gps, original.removed_gps)
        assert restored.livetime_s == 1234.5
        assert restored.n_slides == 17
        assert restored.removal == "hierarchical"
        assert restored.histogram.clustered is True
        assert np.array_equal(restored.histogram.counts, original.histogram.counts)
        assert restored.histogram.overflow == original.histogram.overflow
        queries = np.linspace(-1.0, 45.0, 200)
        assert [restored.n_above(q) for q in queries] == [
            original.n_above(q) for q in queries
        ]

    def test_round_trip_without_optional_arrays(self, tmp_path):
        """A set carrying only statistics survives the trip and stays countable."""
        original = BackgroundSet(
            stats=np.array([3.0, 4.0]),
            livetime_s=99.0,
            n_slides=2,
            removal="inclusive",
        )
        path = tmp_path / "bg_inclusive.h5"
        original.save(path)
        restored = BackgroundSet.load(path)
        assert restored.gps is None
        assert restored.removed_gps is None
        assert restored.histogram is None
        assert restored.n_above(3.5) == 1

    def test_save_creates_parent_directory(self, tmp_path):
        """The stage writes into a run tree it may be the first to reach."""
        path = tmp_path / "run" / "background" / "bg_exclusive.h5"
        BackgroundSet(
            stats=np.array([1.0]),
            livetime_s=10.0,
            n_slides=1,
            removal="exclusive",
        ).save(path)
        assert path.is_file()

    def test_load_missing_file_refused(self, tmp_path):
        """A missing background is reported, not treated as an empty one."""
        with pytest.raises(FileNotFoundError, match="no background set"):
            BackgroundSet.load(tmp_path / "absent.h5")

    @pytest.mark.parametrize("attr", ["livetime_s", "n_slides", "removal"])
    def test_load_requires_every_attribute(self, tmp_path, attr):
        """
        A set missing its livetime or its removal mode cannot be quoted.

        The first is the denominator of every rate and the second says which of three
        backgrounds the number came from; a default for either would be believed.
        """
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "bg.h5"
        BackgroundSet(
            stats=np.array([1.0]),
            livetime_s=10.0,
            n_slides=1,
            removal="inclusive",
        ).save(path)
        with h5py.File(path, "a") as handle:
            del handle.attrs[attr]
        with pytest.raises(ValueError, match=attr):
            BackgroundSet.load(path)

    def test_load_requires_stats_dataset(self, tmp_path):
        """
        A file without its statistics was truncated part-way through a write.

        Reading it as an empty background would report a search that measured no noise
        at all, which is the direction that makes every candidate look significant.
        """
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "bg.h5"
        BackgroundSet(
            stats=np.array([1.0]),
            livetime_s=10.0,
            n_slides=1,
            removal="inclusive",
        ).save(path)
        with h5py.File(path, "a") as handle:
            del handle["stats"]
        with pytest.raises(ValueError, match="stats"):
            BackgroundSet.load(path)


class TestCollateSlides:
    """Accumulating the inclusive background from the slide shards."""

    def test_slides_clustered_independently(self, tmp_path):
        """
        Two triggers at the same time in different slides are two events.

        The fixture is built so the two answers differ by construction: clustered
        together, the loudest trigger of slide 1 wins the whole window and slide 2
        contributes nothing; clustered per slide, each slide keeps its own peak. Letting
        slides suppress one another deletes background that was never coincident, which
        lowers the count and with it every FAR taken from it.
        """
        first = _write_shard(
            tmp_path / "slide1.h5",
            gps=[T0 + 100.0, T0 + 100.5],
            stat=[10.0, 3.0],
            slide_id=[1, 1],
        )
        second = _write_shard(
            tmp_path / "slide2.h5",
            gps=[T0 + 100.25, T0 + 100.75],
            stat=[8.0, 2.0],
            slide_id=[2, 2],
        )
        background = collate_slides([first, second], _plan(), cluster_window_s=1.0)

        assert background.stats.size == 2
        assert np.sort(background.stats).tolist() == [8.0, 10.0]
        assert np.allclose(np.sort(background.gps), [T0 + 100.0, T0 + 100.25])

    def test_livetime_is_the_slide_sum(self, tmp_path):
        """
        Background time is summed from the plan, never inferred from a slide count.

        Per-slide retention falls with lag, so ``n_slides * T_zerolag`` always
        overstates -- and an overstated denominator divides the false-alarm count by too
        much, reporting every rate too low. The fixture's slides retain 900 and 800 s
        against a 1000 s zero lag, so no multiple of the foreground time reproduces it.
        """
        shard = _write_shard(
            tmp_path / "slide1.h5",
            gps=[T0 + 10.0, T0 + 400.0],
            stat=[6.0, 7.0],
            slide_id=[1, 2],
        )
        background = collate_slides([shard], _plan(), cluster_window_s=1.0)

        assert background.livetime_s == pytest.approx(sum(SLID_LIVETIMES_S), abs=1e-12)
        assert background.livetime_s != pytest.approx(
            len(SLID_LIVETIMES_S) * ZEROLAG_LIVETIME_S
        )
        assert background.livetime_s < len(SLID_LIVETIMES_S) * ZEROLAG_LIVETIME_S
        assert background.n_slides == len(SLID_LIVETIMES_S)

    def test_zero_lag_shard_dropped(self, tmp_path):
        """
        Zero-lag triggers among the shards are dropped, not counted.

        The denominator excludes the zero-lag slide, so the numerator has to as well or
        the ratio is not a rate. A shard directory that happens to hold the zero-lag
        shard beside the slides must give the same background as one that does not.
        """
        slid = _write_shard(
            tmp_path / "slide1.h5",
            gps=[T0 + 10.0, T0 + 400.0],
            stat=[6.0, 7.0],
            slide_id=[1, 2],
        )
        zerolag = _write_shard(
            tmp_path / "slide0.h5",
            gps=[T0 + 800.0],
            stat=[99.0],
            slide_id=[0],
        )
        with_foreground = collate_slides([slid, zerolag], _plan(), cluster_window_s=1.0)
        without = collate_slides([slid], _plan(), cluster_window_s=1.0)

        assert np.array_equal(with_foreground.stats, without.stats)
        assert 99.0 not in with_foreground.stats.tolist()

    def test_unknown_slide_refused(self, tmp_path):
        """
        A slide the plan never measured has no livetime behind it.

        Its events would enter a numerator whose denominator never included them, which
        inflates the count against a fixed background time and raises every FAR quoted
        from the campaign.
        """
        shard = _write_shard(
            tmp_path / "slide9.h5", gps=[T0 + 10.0], stat=[6.0], slide_id=[9]
        )
        with pytest.raises(ValueError, match="which the plan does not describe"):
            collate_slides([shard], _plan(), cluster_window_s=1.0)

    def test_missing_slide_column_refused(self, tmp_path):
        """
        Without a slide id the shards cannot be clustered per slide at all.

        Collating them anyway would silently fall back to one clustering across the
        whole ladder, which is the failure this stage exists to prevent.
        """
        shard = _write_shard(
            tmp_path / "noslide.h5", gps=[T0 + 10.0, T0 + 40.0], stat=[6.0, 7.0]
        )
        with pytest.raises(ValueError, match="slide_id"):
            collate_slides([shard], _plan(), cluster_window_s=1.0)

    def test_result_is_marked_clustered(self, tmp_path):
        """
        The collated set declares itself clustered, so it may be counted.

        The histogram totals the survivors and not the raw triggers; a mismatch would
        mean the summary and the statistics describe different amounts of noise.
        """
        shard = _write_shard(
            tmp_path / "slide1.h5",
            gps=[T0 + 10.0, T0 + 10.5, T0 + 400.0],
            stat=[6.0, 5.0, 7.0],
            slide_id=[1, 1, 2],
        )
        background = collate_slides([shard], _plan(), cluster_window_s=1.0)
        assert background.removal == "inclusive"
        assert background.histogram.clustered is True
        assert background.histogram.total == background.stats.size
        assert background.stats.size == 2

    def test_events_keep_their_own_times(self, tmp_path):
        """
        Each surviving statistic is paired with the time of the trigger it came from.

        The removal stages decide coincidence against these times, so a time recomputed
        from the cluster -- a midpoint, a mean -- would veto around an event that never
        existed.
        """
        gps = [T0 + 10.0, T0 + 10.5, T0 + 400.0, T0 + 400.25]
        stat = [6.0, 5.0, 7.0, 9.0]
        shard = _write_shard(
            tmp_path / "slide1.h5", gps=gps, stat=stat, slide_id=[1, 1, 2, 2]
        )
        background = collate_slides([shard], _plan(), cluster_window_s=1.0)
        assert background.gps.shape == background.stats.shape
        pairs = set(zip(background.gps.tolist(), background.stats.tolist()))
        assert pairs <= set(zip(np.asarray(gps, float).tolist(), stat))
        assert pairs == {(T0 + 10.0, 6.0), (T0 + 400.25, 9.0)}


class TestExclusiveBackground:
    """Vetoing on every zero-lag trigger, with the livetime reduced to match."""

    def test_coincident_events_removed(self):
        """
        Background events sharing a zero-lag candidate's detector time are dropped.

        Both families go: the three at ``LOUD_GPS`` in the reference frame, and the three
        that reach ``LOUD_GPS`` only once L1's lag is applied.
        """
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        exclusive = exclusive_background(
            inclusive,
            _real_zerolag([9.0], [LOUD_GPS]),
            plan,
            REAL_GEOMETRY,
            _real_network(),
        )

        assert exclusive.stats.size == inclusive.stats.size - 6
        assert not np.any(np.abs(exclusive.gps - LOUD_GPS) < 1.0)
        assert exclusive.removal == "exclusive"

    def test_livetime_is_reduced(self):
        """
        The denominator loses the time the numerator lost.

        This is the whole correction. Thinning the count while the background time still
        describes the seconds those events were counted in reports every rate low by the
        vetoed fraction -- the direction that makes candidates look more significant than
        the background supports. PyCBC reduces its ``background_time_exc`` for the same
        reason.
        """
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        exclusive = exclusive_background(
            inclusive,
            _real_zerolag([9.0], [LOUD_GPS]),
            plan,
            REAL_GEOMETRY,
            _real_network(),
        )

        assert exclusive.livetime_s < inclusive.livetime_s

    def test_rate_is_not_biased_low(self):
        """
        The exclusive rate exceeds what the unreduced denominator would have reported.

        Stated as the comparison that matters rather than as an equality: the reduced
        livetime can only raise the rate, and the size of the difference is the bias the
        old convention carried.
        """
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        exclusive = exclusive_background(
            inclusive,
            _real_zerolag([9.0], [LOUD_GPS]),
            plan,
            REAL_GEOMETRY,
            _real_network(),
        )
        probe = np.array([6.0])
        honest = far_of_stat(probe, exclusive.stats, exclusive.livetime_s)[0]
        biased = far_of_stat(probe, exclusive.stats, inclusive.livetime_s)[0]

        assert honest > biased

    def test_never_exceeds_the_inclusive_count(self):
        """Removal only removes: at every statistic the exclusive count is no larger."""
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        exclusive = exclusive_background(
            inclusive,
            _real_zerolag([9.0, 9.0], [LOUD_GPS, T0 + 3000.0]),
            plan,
            REAL_GEOMETRY,
            _real_network(),
        )
        for probe in (3.0, 6.0, 9.0, 12.0):
            assert exclusive.n_above(probe) <= inclusive.n_above(probe)

    def test_follower_frame_copies_removed(self):
        """
        Contamination is caught in every detector's frame, not the reference frame alone.

        A real Livingston signal contributes one background copy per slide, and each sits
        at a *different* reference time -- ``LOUD_GPS - offset_L1`` -- so a test made on
        the reference frame sees one of them at most. PyCBC loops the veto over every ifo
        for this reason. In HL a reference-frame test catches half the contamination; in
        HLV, one frame of three.
        """
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        offsets = {int(s.slide_id): dict(s.offsets_s) for s in plan}
        slid = [s.slide_id for s in plan if s.slide_id != 0][:3]
        follower = np.array(
            [LOUD_GPS - float(offsets[int(k)].get("L1", 0.0)) for k in slid]
        )
        exclusive = exclusive_background(
            inclusive,
            _real_zerolag([9.0], [LOUD_GPS]),
            plan,
            REAL_GEOMETRY,
            _real_network(),
        )

        assert follower.size == 3
        assert len(set(np.round(follower, 6))) == 3
        for time in follower:
            # Absolute, not np.isclose: a default rtol on a GPS time is hours wide.
            assert not np.any(np.abs(exclusive.gps - time) < 1e-6)

    def test_window_width_is_the_veto_window(self):
        """
        The window is the constant passed in, applied either side of the candidate.

        An event just inside it goes and one just outside stays. The default is PyCBC's
        ``--veto-window``; no reference scales this by a per-trigger timing uncertainty,
        and Sage's networks report a ``tc_sigma`` two orders of magnitude below it.
        """
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        # Between two noise events, so the narrow window reaches neither.
        near = T0 + 20.0 + 3.0 * 31.0 + 15.0
        wide = exclusive_background(
            inclusive, _real_zerolag([9.0], [near]),
            plan, REAL_GEOMETRY, _real_network(), window_s=20.0,
        )
        narrow = exclusive_background(
            inclusive, _real_zerolag([9.0], [near]),
            plan, REAL_GEOMETRY, _real_network(), window_s=0.001,
        )

        assert wide.stats.size < narrow.stats.size

    def test_mismatched_plan_refused(self):
        """
        The plan must be the ladder the background was collated on.

        A different ladder supplies lags those events were never slid by, so every event
        is placed at a detector time it never occupied and the veto lands on the wrong
        stretches -- silently, since the arithmetic is identical either way.
        """
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        with pytest.raises(ValueError, match="ladder"):
            exclusive_background(
                inclusive,
                _real_zerolag([9.0], [LOUD_GPS]),
                _real_plan(n_slides=6),
                REAL_GEOMETRY,
                _real_network(),
            )

    def test_foreground_livetime_reported(self):
        """
        The reduced zero-lag exposure is carried out, not discarded.

        The veto removes time from the foreground as well as the background, so a FAR
        curve drawn in exclusive mode needs this rather than the inclusive plan's
        foreground time, which describes an exposure the removal ended.
        """
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        exclusive = exclusive_background(
            inclusive,
            _real_zerolag([9.0], [LOUD_GPS]),
            plan,
            REAL_GEOMETRY,
            _real_network(),
        )

        assert exclusive.foreground_livetime_s is not None
        assert exclusive.foreground_livetime_s < plan.foreground_livetime_s

    def test_requires_event_times(self):
        """A background with no times cannot be tested for coincidence against anything."""
        plan = _real_plan()
        bare = _real_inclusive(plan)
        bare.gps = None
        with pytest.raises(ValueError, match="no event times"):
            exclusive_background(
                bare, _real_zerolag([9.0], [LOUD_GPS]),
                plan, REAL_GEOMETRY, _real_network(),
            )

    def test_unclustered_zerolag_refused(self):
        """
        An unclustered zero-lag list vetoes out of all proportion to its events.

        A glitch contributes one trigger per window rather than one candidate, so each of
        those triggers would open its own veto window over the same stretch.
        """
        plan = _real_plan()
        with pytest.raises(ValueError, match="unclustered"):
            exclusive_background(
                _real_inclusive(plan),
                _real_zerolag([9.0], [LOUD_GPS], clustered=False),
                plan,
                REAL_GEOMETRY,
                _real_network(),
            )


class TestHierarchicalRemoval:
    """sgwc-1's louder-than gate, with the data of the removed candidate taken out."""

    def test_louder_candidate_removes_its_family(self):
        """
        A zero-lag candidate louder than the background it contaminates takes it out.

        The three loud background events are one real signal seen once per slide. A
        candidate above them at the same time is evidence they are slid copies of it, not
        noise, so the stretch it occupies leaves every slide at once.
        """
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        result = hierarchical_removal(
            inclusive,
            _real_zerolag([15.0], [LOUD_GPS]),
            plan,
            REAL_GEOMETRY,
            _real_network(),
            min_background_livetime_s=0.0,
        )

        assert result.removal == "hierarchical"
        assert result.stats.max() < 10.0
        assert result.removed_gps.size == 1
        assert result.livetime_s < inclusive.livetime_s

    def test_quieter_candidate_removes_nothing(self):
        """
        A candidate quieter than the background event it coincides with is no evidence.

        This is the gate that separates this from the exclusive background, which vetoes
        on every zero-lag trigger regardless. The noise reached higher there than the
        candidate did, so the background event stands.
        """
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        result = hierarchical_removal(
            inclusive,
            _real_zerolag([4.0], [LOUD_GPS]),
            plan,
            REAL_GEOMETRY,
            _real_network(),
            min_background_livetime_s=0.0,
        )

        assert result.removed_gps.size == 0
        assert result.stats.size == inclusive.stats.size
        assert result.livetime_s == inclusive.livetime_s

    def test_livetime_floor_declines_the_removal(self):
        """
        A removal that would breach the floor is declined, and the walk continues.

        Checked before committing, so the floor is a guarantee about what is kept rather
        than a report of what was lost. Setting the floor at the starting livetime means
        every removal would breach it, so nothing is removed at all.
        """
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        result = hierarchical_removal(
            inclusive,
            _real_zerolag([15.0], [LOUD_GPS]),
            plan,
            REAL_GEOMETRY,
            _real_network(),
            min_background_livetime_s=inclusive.livetime_s,
        )

        assert result.removed_gps.size == 0
        assert result.livetime_s == inclusive.livetime_s

    def test_floor_is_respected_when_it_binds(self):
        """The surviving background never falls below the floor the caller asked for."""
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        floor = 0.995 * inclusive.livetime_s
        result = hierarchical_removal(
            inclusive,
            _real_zerolag([15.0, 15.0, 15.0], [LOUD_GPS, T0 + 400.0, T0 + 900.0]),
            plan,
            REAL_GEOMETRY,
            _real_network(),
            min_background_livetime_s=floor,
        )

        assert result.livetime_s >= floor

    def test_significance_rule_stops_at_the_loudest_survivor(self):
        """
        PyCBC's rule: no candidate outranks the whole surviving background, so stop.

        Under counting FAR that is exactly ``any(ifar_foreground >= background_time)``
        going false. The candidate here is louder than the second and third background
        events but not the first, so nothing outranks the background and the walk ends
        with the background untouched.
        """
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        result = hierarchical_removal(
            inclusive,
            _real_zerolag([11.95], [LOUD_GPS]),
            plan,
            REAL_GEOMETRY,
            _real_network(),
            stop_rule="significance",
        )

        assert result.removed_gps.size == 0
        assert result.stats.max() == pytest.approx(12.0)

    def test_counted_rule_walks_past_a_clean_event(self):
        """
        sgwc-1's rule: a clean background event is stepped over, not stopped on.

        The same candidate that ends the significance walk removes contamination here,
        because the walk continues below the loudest background event and reaches the
        quieter ones the candidate does outrank. This is the difference the two rules
        make, and it is why stopping at the first clean event collapsed the hierarchical
        set onto the inclusive one.
        """
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        result = hierarchical_removal(
            inclusive,
            _real_zerolag([11.95], [LOUD_GPS]),
            plan,
            REAL_GEOMETRY,
            _real_network(),
            stop_rule="counted",
        )

        assert result.removed_gps.size == 1
        assert result.stats.size < inclusive.stats.size
        assert result.livetime_s < inclusive.livetime_s

    def test_counted_rule_honours_ignore_limit(self):
        """The counted walk ends after enough consecutive clean events."""
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        result = hierarchical_removal(
            inclusive,
            _real_zerolag([11.95], [LOUD_GPS]),
            plan,
            REAL_GEOMETRY,
            _real_network(),
            stop_rule="counted",
            ignore_limit=0,
        )

        assert result.removed_gps.size == 0

    def test_default_floor_removes_nothing_from_the_walk(self):
        """
        The default floor is zero, so it never declines a removal.

        A floor defaulting to a year silently turned every campaign at or below a year of
        background -- which is how sgwc-1's own was sized -- into a no-op returning the
        inclusive set under a hierarchical label. Neither reference imposes a floor.
        """
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        zerolag = _real_zerolag([15.0], [LOUD_GPS])
        default = hierarchical_removal(
            inclusive, zerolag, plan, REAL_GEOMETRY, _real_network()
        )
        explicit = hierarchical_removal(
            inclusive, zerolag, plan, REAL_GEOMETRY, _real_network(),
            min_background_livetime_s=0.0,
        )

        assert default.removed_gps.size == explicit.removed_gps.size
        assert default.removed_gps.size > 0
        assert default.livetime_s == explicit.livetime_s

    def test_bad_stop_rule_refused(self):
        """An unknown stopping rule is not silently treated as the default."""
        plan = _real_plan()
        with pytest.raises(ValueError, match="stop_rule"):
            hierarchical_removal(
                _real_inclusive(plan),
                _real_zerolag([15.0], [LOUD_GPS]),
                plan,
                REAL_GEOMETRY,
                _real_network(),
                stop_rule="whatever",
            )

    def test_bracketed_by_the_other_two(self):
        """
        At every statistic the hierarchical count lies between exclusive and inclusive.

        The ordering is structural **at a common window**: hierarchical vetoes on the
        subset of candidates that pass the louder-than gate, exclusive on all of them.
        The two defaults are different numbers -- PyCBC's ``--veto-window`` and
        ``--hierarchical-removal-window`` -- so the window is passed explicitly here.
        """
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        zerolag = _real_zerolag([15.0], [LOUD_GPS])
        network = _real_network()
        window = 1.0
        exclusive = exclusive_background(
            inclusive, zerolag, plan, REAL_GEOMETRY, network, window_s=window
        )
        hierarchical = hierarchical_removal(
            inclusive, zerolag, plan, REAL_GEOMETRY, network, window_s=window
        )
        for probe in (3.0, 5.0, 7.0, 9.0, 11.0, 13.0):
            assert (
                exclusive.n_above(probe)
                <= hierarchical.n_above(probe)
                <= inclusive.n_above(probe)
            )

    def test_respects_max_iterations(self):
        """A hard bound on removals, independent of what the inputs contain."""
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        result = hierarchical_removal(
            inclusive,
            _real_zerolag([15.0, 15.0], [LOUD_GPS, T0 + 400.0]),
            plan,
            REAL_GEOMETRY,
            _real_network(),
            min_background_livetime_s=0.0,
            max_iterations=0,
        )

        assert result.removed_gps.size == 0

    def test_requires_slide_ids(self):
        """
        Without slide ids an event cannot be placed in any detector's own frame.

        Contamination reaches the background through the detector data an event used, and
        under a slide that is not the frame the event is recorded in.
        """
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        inclusive.slide_id = None
        with pytest.raises(ValueError, match="no slide ids"):
            hierarchical_removal(
                inclusive,
                _real_zerolag([15.0], [LOUD_GPS]),
                plan,
                REAL_GEOMETRY,
                _real_network(),
                min_background_livetime_s=0.0,
            )

    def test_bad_floor_refused(self):
        """A livetime floor that is not a non-negative number bounds nothing."""
        plan = _real_plan()
        inclusive = _real_inclusive(plan)
        for bad in (-1.0, np.nan, np.inf):
            with pytest.raises(ValueError, match="min_background_livetime_s"):
                hierarchical_removal(
                    inclusive,
                    _real_zerolag([15.0], [LOUD_GPS]),
                    plan,
                    REAL_GEOMETRY,
                    _real_network(),
                    min_background_livetime_s=bad,
                )

class TestOverdispersion:
    """Poisson against negative binomial on binned background counts."""

    def test_poisson_counts_not_flagged(self):
        """
        Genuinely Poisson counts are reported as Poisson.

        Half the test: an implementation that always reports over-dispersion passes any
        check made only on clustered data, and would condemn every valid background the
        search ever produces.
        """
        counts = np.random.default_rng(3).poisson(20.0, size=400)
        result = overdispersion_lrt(counts)
        assert result["overdispersed"] is False
        assert result["p_value"] > 0.05
        assert result["alpha"] == pytest.approx(0.0, abs=1e-3)
        assert result["index_of_dispersion"] == pytest.approx(1.0, rel=0.2)

    def test_clustered_counts_flagged(self):
        """
        Gamma-mixed Poisson counts are reported as over-dispersed.

        This is what a background whose triggers arrive in bursts looks like: the
        variance is an order of magnitude above the mean, and order-statistic counting
        of the FAR is no longer valid on it.
        """
        rng = np.random.default_rng(4)
        alpha, mean = 0.5, 20.0
        rates = rng.gamma(shape=1.0 / alpha, scale=alpha * mean, size=400)
        result = overdispersion_lrt(rng.poisson(rates))
        assert result["overdispersed"] is True
        assert result["p_value"] < 1e-6
        assert result["index_of_dispersion"] > 5.0

    def test_fitted_alpha_matches_the_draw(self):
        """
        The fitted dispersion recovers the one the counts were drawn with.

        A flag alone would pass for a fit that pinned alpha at either bound; recovering
        the value is what says the profile likelihood is being maximised rather than
        merely evaluated somewhere.
        """
        rng = np.random.default_rng(4)
        alpha, mean = 0.5, 20.0
        rates = rng.gamma(shape=1.0 / alpha, scale=alpha * mean, size=400)
        result = overdispersion_lrt(rng.poisson(rates))
        assert result["alpha"] == pytest.approx(alpha, rel=0.3)
        assert result["mean"] == pytest.approx(mean, rel=0.2)
        assert result["loglik_negbin"] > result["loglik_poisson"]

    def test_false_positive_rate_near_the_level(self):
        """
        Independent Poisson backgrounds are flagged at about the declared level.

        The single-sample test above passes for a statistic that is merely noisy;
        forty replicates bound how often a valid background would be condemned, which
        is the operational cost of the test being wrong in that direction.
        """
        flagged = sum(
            overdispersion_lrt(
                np.random.default_rng(1000 + seed).poisson(15.0, size=300)
            )["overdispersed"]
            for seed in range(40)
        )
        assert flagged <= 6

    def test_boundary_halves_the_p_value(self):
        """
        The p-value is half the chi-square tail, because alpha cannot be negative.

        The null puts half its mass at the boundary, so the naive one-degree-of-freedom
        tail reports every background as twice as Poisson-like as it is -- in the
        direction that lets an over-dispersed background pass unremarked.
        """
        chi2 = pytest.importorskip("scipy.stats").chi2
        rng = np.random.default_rng(13)
        alpha, mean = 0.01, 20.0
        rates = rng.gamma(shape=1.0 / alpha, scale=alpha * mean, size=300)
        result = overdispersion_lrt(rng.poisson(rates))
        assert result["statistic"] > 1.0
        assert result["p_value"] == pytest.approx(
            0.5 * chi2.sf(result["statistic"], 1), rel=1e-9
        )

    def test_underdispersed_reports_the_boundary(self):
        """
        Counts tighter than Poisson sit at the boundary and report exactly zero.

        A likelihood ratio in favour of the nested model is a rounding artefact, and
        reporting the small negative value the arithmetic produces would let a p-value
        above one reach a table.
        """
        result = overdispersion_lrt(np.full(50, 7, dtype=np.int64))
        assert result["alpha"] == 0.0
        assert result["statistic"] == 0.0
        assert result["p_value"] == 1.0
        assert result["overdispersed"] is False
        assert result["loglik_negbin"] == result["loglik_poisson"]

    @pytest.mark.parametrize(
        "counts,message",
        [
            (np.array([[1, 2], [3, 4]]), "one-dimensional"),
            (np.array([3]), "at least two bins"),
            (np.array([1.5, 2.5]), "whole numbers"),
            (np.array([-1, 2]), "whole numbers"),
            (np.array([0, 0, 0]), "every bin is empty"),
            (np.array([1.0, np.nan]), "non-finite"),
        ],
    )
    def test_invalid_counts_refused(self, counts, message):
        """
        Inputs with no Poisson likelihood are refused rather than fitted.

        A fractional count is a rate or a weight, an empty set has no dispersion to
        measure, and a single bin has nothing to compare against -- each would return a
        plausible-looking dispersion that describes nothing.
        """
        with pytest.raises(ValueError, match=message):
            overdispersion_lrt(counts)


class TestShardCoverage:
    """Collation must account for every slide the plan describes."""

    def test_missing_slide_refused(self, tmp_path):
        """
        A slide with no shard at all is refused, not silently dropped.

        The plan's livetime fills the denominator whether or not a slide's job ran, so a
        preempted or crashed slide leaves the numerator short against the full background
        time and reports every rate low -- the direction that makes candidates look more
        significant. Nothing else in the pipeline would notice.
        """
        shard = _write_shard(
            tmp_path / "slide1.h5",
            gps=[T0 + 10.0, T0 + 400.0],
            stat=[6.0, 7.0],
            slide_id=[1, 1],
        )
        with pytest.raises(ValueError, match="produced no shard"):
            collate_slides([shard], _plan(), cluster_window_s=1.0)

    def test_empty_slide_accepted(self, tmp_path):
        """
        A slide that ran and found nothing is ordinary and must still collate.

        It is distinguishable from a missing slide only by its declared ``slide_id``
        attribute: both contribute no rows. Its livetime is real exposure and belongs in
        the denominator, so refusing it would be as wrong as ignoring a lost one.
        """
        occupied = _write_shard(
            tmp_path / "slide1.h5",
            gps=[T0 + 10.0, T0 + 400.0],
            stat=[6.0, 7.0],
            slide_id=[1, 1],
        )
        empty = tmp_path / "slide2.h5"
        writer = TriggerWriter(empty, dict(_PROVENANCE, slide_id=2))
        writer.add_histogram(
            histogram_stats(np.asarray([], dtype=np.float64), clustered=False)
        )
        writer.complete_block(0)
        writer.close()

        background = collate_slides([occupied, empty], _plan(), cluster_window_s=1.0)

        assert background.stats.size == 2
        assert background.livetime_s == pytest.approx(sum(SLID_LIVETIMES_S), abs=1e-12)

    def test_linkage_forwarded(self, tmp_path):
        """
        The ``linkage`` argument reaches the clusterer rather than being ignored.

        Three triggers 0.6 s apart under a 1 s window separate the two rules: the peak
        rule keeps both ends, since neither is outranked inside its own window, while gap
        linkage chains all three into one cluster and keeps only the loudest. A
        pass-through that dropped the argument would report the peak count for both.
        """
        shard = _write_shard(
            tmp_path / "slides.h5",
            gps=[T0, T0 + 0.6, T0 + 1.2, T0 + 500.0],
            stat=[5.0, 3.0, 6.0, 4.0],
            slide_id=[1, 1, 1, 2],
        )
        peak = collate_slides([shard], _plan(), cluster_window_s=1.0, linkage="peak")
        gap = collate_slides([shard], _plan(), cluster_window_s=1.0, linkage="gap")

        assert np.sort(peak.stats).tolist() == [4.0, 5.0, 6.0]
        assert np.sort(gap.stats).tolist() == [4.0, 6.0]
