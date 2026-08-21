#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_cluster.py
Description   : Trigger clustering, including the cases that change counts.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Clustering sets the event count that every rate is divided by, so its edge cases are
worth pinning precisely.
"""

import numpy as np
import pytest

from sage.search.cluster import (
    cluster_slides,
    cluster_triggers,
    cluster_with_halo,
    coincidence_time,
    group_events,
)

WINDOW_S = 0.35   # the reference pipeline's trigger-level window


def _bumpy_train():
    """
    An unbroken 9.5 s trigger train carrying five separated maxima.

    The shape a glitch train has: no gap anywhere wide enough to break a chain, but five
    genuinely distinct events in it. It ends on a trough so the fifth bump is the last
    one and the count is exactly five rather than five and a truncated sixth.
    """
    n = 191                                        # 0 .. 190, ending at a minimum
    times = np.arange(n) * 0.05                    # 9.5 s at 0.05 s, never a gap
    stats = 5.0 + np.sin(np.arange(n) * (2 * np.pi / 40))
    return times, stats


def _reference_clusters(times, stats, threshold):
    """
    ``benchmark/mlgwsc1/mlgwsc1.py::get_clusters``, transcribed without its printing.

    Kept verbatim in behaviour so ``linkage="gap"`` can be checked against the pipeline
    it exists to be comparable with, rather than against a paraphrase of it.
    """
    clusters = []
    for time, stat in zip(times, stats):
        if not clusters or (time - clusters[-1][-1][0]) > threshold:
            clusters.append([(time, stat)])
        else:
            clusters[-1].append((time, stat))
    out_t, out_v = [], []
    for cluster in clusters:
        values = np.array([entry[1] for entry in cluster])
        best = int(np.argmax(values))
        out_t.append(cluster[best][0])
        out_v.append(cluster[best][1])
    return np.array(out_t), np.array(out_v)


class TestBasics:
    """Behaviour on small, hand-checkable inputs."""

    def test_empty_input(self):
        """No triggers gives no clusters."""
        result = cluster_triggers(np.array([]), np.array([]), WINDOW_S)
        assert len(result) == 0
        assert result.rep_index.size == 0
        assert result.times.size == 0
        # The empty case answers in the same dtypes as a populated one, so a caller
        # concatenating shards does not get an object array from the empty shard.
        assert result.rep_index.dtype == np.int64
        assert result.stats.dtype == np.float64

    def test_single_trigger(self):
        """One trigger is its own cluster."""
        result = cluster_triggers(np.array([10.0]), np.array([8.5]), WINDOW_S)
        assert len(result) == 1
        assert result.rep_index.tolist() == [0]
        assert result.times.tolist() == [10.0]
        assert result.stats.tolist() == [8.5]
        assert result.size.tolist() == [1]
        assert result.t0.tolist() == [10.0] == result.t1.tolist()

    def test_isolated_stay_separate(self):
        """N triggers spaced beyond the window give exactly N clusters."""
        times = np.arange(6) * (10 * WINDOW_S)
        stats = np.array([1.0, 5.0, 2.0, 9.0, 3.0, 4.0])
        result = cluster_triggers(times, stats, WINDOW_S)
        assert len(result) == 6
        assert result.rep_index.tolist() == list(range(6))
        assert result.size.tolist() == [1] * 6

    def test_loudest_represents(self):
        """The surviving trigger is the highest ranked in its cluster."""
        times = np.array([0.0, 0.1, 0.2, 0.3])
        stats = np.array([2.0, 7.0, 3.0, 1.0])
        result = cluster_triggers(times, stats, WINDOW_S)
        assert len(result) == 1
        assert result.rep_index.tolist() == [1]
        assert result.stats.tolist() == [7.0]
        # The extent covers the triggers that were discarded, not just the survivor.
        assert result.t0.tolist() == [0.0]
        assert result.t1.tolist() == [0.3]
        assert result.size.tolist() == [4]

    def test_exact_ties(self):
        """
        Equal statistics resolve deterministically.

        A ranking statistic quantised by fp16 storage produces exact ties often, and a
        rule that depended on sort stability or on iteration order would give a different
        candidate list on a re-run of the same data.
        """
        times = np.array([0.0, 0.1, 0.2])
        stats = np.array([5.0, 5.0, 5.0])
        result = cluster_triggers(times, stats, WINDOW_S)
        assert result.rep_index.tolist() == [0], "the earliest of equal statistics wins"

        # And the rule does not depend on where the equal run sits.
        stats = np.array([1.0, 5.0, 5.0])
        assert cluster_triggers(times, stats, WINDOW_S).rep_index.tolist() == [1]

    def test_window_boundary(self):
        """
        Behaviour at the boundary is defined and consistent on both sides.

        The test is inclusive: exactly one window apart is the same cluster. Which way it
        falls matters less than that it is decided, but it has to match the reference
        pipeline's `>` test or the two cannot be compared trigger for trigger.
        """
        stats = np.array([9.0, 1.0])
        at = cluster_triggers(np.array([0.0, WINDOW_S]), stats, WINDOW_S)
        assert len(at) == 1

        just_over = np.nextafter(WINDOW_S, 1.0)
        beyond = cluster_triggers(np.array([0.0, just_over]), stats, WINDOW_S)
        assert len(beyond) == 2

        just_under = np.nextafter(WINDOW_S, 0.0)
        within = cluster_triggers(np.array([0.0, just_under]), stats, WINDOW_S)
        assert len(within) == 1

    def test_unsorted_refused(self):
        """
        A shuffled train is refused rather than clustered into nonsense.

        Every routine here assumes ascending time -- the window bounds are found by
        binary search -- and a shuffled input silently produces clusters that are not
        contiguous in time, which no downstream count would flag.
        """
        with pytest.raises(ValueError, match="ascending"):
            cluster_triggers(np.array([1.0, 0.0]), np.array([1.0, 2.0]), WINDOW_S)

    def test_length_mismatch_refused(self):
        """Times and statistics describe the same triggers or the pairing is lost."""
        with pytest.raises(ValueError, match="same length"):
            cluster_triggers(np.array([0.0, 1.0]), np.array([1.0]), WINDOW_S)


class TestLinkage:
    """The two linkage rules differ where a train is continuous."""

    def test_peak_bounds_extent(self):
        """
        Anchoring on the loudest keeps a cluster within one window of its peak.

        A continuous 10 s train carrying five separated bumps -- a glitch train with
        several genuine maxima in it -- must yield those five events. A rule that chained
        through the train would report one, hiding every event after the first; that is
        the failure the peak rule exists to prevent, and it is why the count here is
        asserted exactly rather than as "more than one".

        Every discarded trigger is within one window of a trigger that outranked it,
        which is the bound stated in the module docstring, checked directly.
        """
        times, stats = _bumpy_train()
        result = cluster_triggers(times, stats, WINDOW_S, linkage="peak")

        assert len(result) == 5, "five bumps are five events"
        survivors = set(result.rep_index.tolist())
        for index in range(times.size):
            if index in survivors:
                continue
            near = np.abs(times - times[index]) <= WINDOW_S
            assert stats[near].max() > stats[index] or np.flatnonzero(
                near & (stats == stats[near].max())
            )[0] < index, "a discarded trigger was beaten inside one window"

    def test_extents_partition_train(self):
        """
        Every trigger belongs to exactly one cluster, so the sizes sum to the input.

        The extents describe the discarded triggers as well as the survivor -- that is
        what says how long the train behind a candidate ran -- so each trigger is
        assigned to its nearest representative and none is dropped from the accounting.
        """
        rng = np.random.default_rng(5)
        times = np.sort(rng.uniform(0.0, 100.0, size=800))
        stats = rng.normal(size=800)
        result = cluster_triggers(times, stats, WINDOW_S)

        assert int(result.size.sum()) == times.size == result.n_triggers
        assert np.all(result.t0 <= result.times + 1e-12)
        assert np.all(result.t1 >= result.times - 1e-12)

    def test_peak_survivors_are_local_maxima(self):
        """
        Every representative outranks everything within a window of it, and only those do.

        This is the clustering rule stated in prose, asserted directly rather than
        through its consequences, and it is what makes blockwise clustering exact.
        """
        rng = np.random.default_rng(7)
        times = np.sort(rng.uniform(0.0, 50.0, size=400))
        stats = rng.normal(size=400)
        result = cluster_triggers(times, stats, WINDOW_S, linkage="peak")

        survivors = set(result.rep_index.tolist())
        for index in range(times.size):
            near = np.flatnonzero(np.abs(times - times[index]) <= WINDOW_S)
            best = near[int(np.argmax(stats[near]))]  # argmax takes the earliest tie
            assert (index in survivors) == (best == index)

    def test_gap_chains_dense_train(self):
        """Anchoring on the last trigger allows a cluster to extend indefinitely."""
        times, stats = _bumpy_train()
        gap = cluster_triggers(times, stats, WINDOW_S, linkage="gap")
        peak = cluster_triggers(times, stats, WINDOW_S, linkage="peak")

        assert len(gap) == 1, "an unbroken train chains into a single cluster"
        assert len(peak) == 5, "the same train holds five events"
        assert gap.t1[0] - gap.t0[0] == pytest.approx(times[-1] - times[0])

    def test_gap_matches_mlgwsc1(self):
        """
        ``linkage="gap"`` is the reference implementation, trigger for trigger.

        It is kept only so the engine can be compared against
        ``benchmark/mlgwsc1/mlgwsc1.py::get_clusters`` on the same data. If it drifts, a
        disagreement in the comparison would be read as a difference in the network when
        it was a difference in the clustering.
        """
        rng = np.random.default_rng(11)
        times = np.sort(rng.uniform(0.0, 200.0, size=2000))
        stats = rng.normal(size=2000)
        result = cluster_triggers(times, stats, WINDOW_S, linkage="gap")
        want_t, want_v = _reference_clusters(times, stats, WINDOW_S)

        assert result.times.tolist() == want_t.tolist()
        assert result.stats.tolist() == want_v.tolist()

    def test_peak_matches_pycbc(self):
        """
        The default rule is PyCBC's, checked against PyCBC rather than against a copy.

        ``pycbc.events.coinc.cluster_over_time`` is the production implementation of
        exactly this rule, so it is the strongest oracle available: an independent, long
        exercised implementation rather than a transcription that could share a mistake
        with the code under test. Both of its methods are checked, since ours borrows the
        skip-ahead structure of the python one.

        Times are drawn continuously so no pair lands exactly one window apart, which is
        the one case where the two conventions deliberately differ -- see
        ``test_window_edge_differs_from_pycbc``.
        """
        coinc = pytest.importorskip("pycbc.events.coinc")
        rng = np.random.default_rng(23)
        for size, span in ((500, 50.0), (5000, 200.0), (20000, 100.0)):
            times = np.sort(rng.uniform(0.0, span, size=size))
            stats = rng.normal(size=size)
            ours = cluster_triggers(times, stats, WINDOW_S).rep_index

            for method in ("python", "cython"):
                try:
                    theirs = coinc.cluster_over_time(
                        stats, times, WINDOW_S, method=method
                    )
                except (ImportError, ValueError):  # pragma: no cover - optional cython
                    continue
                assert ours.tolist() == np.sort(theirs).tolist(), (size, method)

    def test_window_edge_differs_from_pycbc(self):
        """
        Two triggers exactly one window apart are one cluster here, two in PyCBC.

        PyCBC takes both window edges with the default ``searchsorted`` side, so its
        window is ``[t - w, t + w)``: the earlier trigger cannot see the later one, but
        the later can see the earlier. When the later is the louder, both survive -- two
        events exactly one clustering window apart, from a rule meant to leave one.

        Ours is closed on both sides, which also makes it agree with the gap linkage's
        ``>`` test at the same boundary. The difference is invisible on continuous times
        and deliberate on a lattice, where a window commensurate with the sample spacing
        makes exact hits systematic rather than rare.
        """
        coinc = pytest.importorskip("pycbc.events.coinc")
        times = np.array([0.0, WINDOW_S])
        stats = np.array([1.0, 9.0])   # the later one is louder

        assert cluster_triggers(times, stats, WINDOW_S).rep_index.tolist() == [1]
        assert sorted(coinc.cluster_over_time(stats, times, WINDOW_S).tolist()) == [0, 1]

    def test_unknown_linkage_refused(self):
        """A misspelled rule must not fall back to a default and change the count."""
        with pytest.raises(ValueError, match="linkage"):
            cluster_triggers(np.array([0.0]), np.array([1.0]), WINDOW_S, linkage="single")

    def test_payload_follows_representative(self):
        """
        Extra columns are carried through by representative index.

        A candidate's chirp mass and its statistic have to come from the same trigger.
        Anything re-derived across the cluster -- a mean, a midpoint -- would describe a
        trigger that does not exist.
        """
        times = np.array([0.0, 0.1, 0.2, 5.0])
        stats = np.array([2.0, 7.0, 3.0, 4.0])
        mchirp = np.array([20.0, 31.5, 22.0, 40.0])
        segment = np.array([4, 4, 4, 9])
        result = cluster_triggers(
            times, stats, WINDOW_S, payload={"mchirp": mchirp, "segment_index": segment}
        )
        assert result.columns["mchirp"].tolist() == [31.5, 40.0]
        assert result.columns["segment_index"].tolist() == [4, 9]
        # And the same through the explicit call, for a column produced later.
        assert result.payload({"mchirp": mchirp})["mchirp"].tolist() == [31.5, 40.0]

    def test_payload_length_checked(self):
        """
        A column that is not one row per trigger is refused, not indexed anyway.

        `rep_index` points into the trigger train, so a shorter or longer column is read
        at the wrong position and attaches another trigger's parameters to a candidate --
        silently, and the numbers stay plausible.
        """
        times = np.array([0.0, 0.1, 5.0])
        stats = np.array([2.0, 7.0, 3.0])
        result = cluster_triggers(times, stats, WINDOW_S)
        with pytest.raises(ValueError, match="rows against"):
            result.payload({"mchirp": np.array([1.0, 2.0])})


class TestBlockBoundaries:
    """A cluster spanning a boundary must be emitted once."""

    def test_straddling_cluster_not_split(self):
        """
        A single cluster crossing a block edge yields one representative.

        Splitting it would add one background event per boundary, biasing the count
        upward in the direction that inflates significance.
        """
        times = np.array([9.9, 9.95, 10.0, 10.05, 10.1])
        stats = np.array([1.0, 2.0, 9.0, 2.5, 1.5])
        # The cluster's peak sits at 10.0, exactly on the boundary between the blocks.
        first = cluster_with_halo(times, stats, WINDOW_S, 0.0, 10.0, halo_s=1.0)
        second = cluster_with_halo(times, stats, WINDOW_S, 10.0, 20.0, halo_s=1.0)

        assert len(first) == 0
        assert len(second) == 1
        assert second.times.tolist() == [10.0]
        assert len(first) + len(second) == len(cluster_triggers(times, stats, WINDOW_S))

    def test_halo_emits_once(self):
        """A representative in the preceding halo is dropped, not emitted twice."""
        times = np.array([5.0, 9.5, 12.0, 15.0, 21.0])
        stats = np.array([3.0, 4.0, 8.0, 6.0, 2.0])
        blocks = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
        emitted = [
            cluster_with_halo(times, stats, WINDOW_S, lo, hi, halo_s=1.0).times.tolist()
            for lo, hi in blocks
        ]
        flattened = [time for block in emitted for time in block]
        assert flattened == sorted(flattened)
        assert len(flattened) == len(set(flattened)), "no trigger is emitted twice"
        assert flattened == cluster_triggers(times, stats, WINDOW_S).times.tolist()

    def test_blockwise_equals_wholesale(self):
        """
        Clustering in blocks with a halo matches clustering the whole set at once.

        The background is clustered by many jobs in parallel, so this is the property
        that makes the parallel answer the same as the serial one. It is asserted
        exactly, not approximately: one extra or one missing background event changes
        every FAR that divides by the count.
        """
        rng = np.random.default_rng(3)
        times = np.sort(rng.uniform(0.0, 600.0, size=5000))
        stats = rng.normal(size=5000)

        whole = cluster_triggers(times, stats, WINDOW_S)
        edges = np.arange(0.0, 660.0, 60.0)
        pieces = [
            cluster_with_halo(times, stats, WINDOW_S, lo, hi, halo_s=WINDOW_S)
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
        assert np.concatenate([p.times for p in pieces]).tolist() == whole.times.tolist()
        assert np.concatenate([p.stats for p in pieces]).tolist() == whole.stats.tolist()
        assert (
            np.concatenate([p.rep_index for p in pieces]).tolist()
            == whole.rep_index.tolist()
        )

    def test_short_halo_refused(self):
        """
        Too small a halo is an error, not a slightly wrong answer.

        With a halo under one window a trigger just inside the block cannot see
        everything competing with it, so it can survive in its block having been beaten
        by a trigger the block never showed it. That adds background events at every
        boundary and lowers every FAR, and nothing downstream would report it.
        """
        with pytest.raises(ValueError, match="narrower than"):
            cluster_with_halo(
                np.array([1.0]), np.array([1.0]), WINDOW_S, 0.0, 10.0, halo_s=WINDOW_S / 2
            )

    def test_short_halo_changes_count(self):
        """
        The refusal above guards a real difference, not a hypothetical one.

        Built by hand: a quiet trigger sits just inside the block and a louder one just
        outside it, within a window. With a full halo the quiet one is correctly
        discarded; with none it survives and the block reports an event that clustering
        the whole run does not.
        """
        times = np.array([9.8, 10.1])
        stats = np.array([3.0, 9.0])
        assert cluster_triggers(times, stats, WINDOW_S).times.tolist() == [10.1]

        blind = cluster_triggers(times[:1], stats[:1], WINDOW_S)
        assert blind.times.tolist() == [9.8], "without the halo the quiet one survives"

        seen = cluster_with_halo(times, stats, WINDOW_S, 0.0, 10.0, halo_s=WINDOW_S)
        assert len(seen) == 0


class TestSlides:
    """Each slide is its own realisation of the background."""

    def test_slides_independent(self):
        """
        Two triggers at the same time in different slides both survive.

        They are not the same event: each slide is an independent draw of the background,
        and letting one suppress the other removes background events that were never
        coincident. The count only falls, so every FAR taken from it falls with it.
        """
        times = np.array([100.0, 100.0, 100.01, 100.01])
        stats = np.array([5.0, 9.0, 4.0, 8.0])
        slides = np.array([1, 2, 1, 2])
        result = cluster_slides(times, stats, slides, WINDOW_S)

        assert len(result) == 2
        # One survivor per slide: the loudest of that slide, not of the pair of slides.
        assert sorted(result.stats.tolist()) == [5.0, 9.0]
        assert set(result.rep_index.tolist()) == {0, 1}
        # Clustered together as one train, the 5.0 would have been suppressed by the 9.0.
        flat = cluster_triggers(np.sort(times), stats[np.argsort(times)], WINDOW_S)
        assert len(flat) == 1

    def test_banded_equals_per_slide(self):
        """
        The one-pass banded sweep gives exactly what a loop over slides gives.

        Displacing each slide into a band of its own is an optimisation, so it has to be
        indistinguishable from the obvious implementation -- asserted exactly, since one
        background event either way moves every rate.
        """
        rng = np.random.default_rng(29)
        n = 4000
        times = rng.uniform(0.0, 500.0, size=n)
        stats = rng.normal(size=n)
        slides = rng.integers(0, 20, size=n)

        together = cluster_slides(times, stats, slides, WINDOW_S)

        separate = []
        for slide in range(20):
            where = np.flatnonzero(slides == slide)
            where = where[np.argsort(times[where], kind="stable")]
            inner = cluster_triggers(times[where], stats[where], WINDOW_S)
            separate.extend(where[inner.rep_index].tolist())

        assert sorted(together.rep_index.tolist()) == sorted(separate)

    def test_zero_lag_clusters_normally(self):
        """Slide 0 clusters like any other; it is the FAR layer that treats it apart."""
        times = np.array([10.0, 10.0])
        stats = np.array([3.0, 4.0])
        result = cluster_slides(times, stats, np.array([0, 7]), WINDOW_S)
        assert len(result) == 2

    def test_extents_undo_banding(self):
        """
        The banding is internal: a cluster's extent comes back on the GPS axis.

        A t0 still carrying its displacement would be off by the band width -- a number
        of order the run length, which would look like a plausible GPS time and place the
        candidate in the wrong part of the run.
        """
        times = np.array([1238166018.0, 1238166018.1, 1238166030.0])
        stats = np.array([2.0, 8.0, 5.0])
        slides = np.array([3, 3, 9])
        result = cluster_slides(times, stats, slides, WINDOW_S)

        assert np.all(result.t0 >= times.min() - 1e-6)
        assert np.all(result.t1 <= times.max() + 1e-6)
        assert result.t0[0] == pytest.approx(1238166018.0, abs=1e-6)
        assert result.t1[0] == pytest.approx(1238166018.1, abs=1e-6)


class TestCoincidenceTime:
    """One time per coincidence, over the detectors that were in it."""

    def test_mean_over_participating(self):
        """A two-detector coincidence in a three-detector campaign averages over two."""
        times, n = coincidence_time(
            {
                "H1": np.array([100.0, 200.0]),
                "L1": np.array([100.01, 200.01]),
                "V1": np.array([-1.0, 200.02]),   # absent from the first
            }
        )
        assert n.tolist() == [2, 3]
        assert times[0] == pytest.approx(100.005)
        assert times[1] == pytest.approx(200.01)

    def test_offset_removed_before_mean(self):
        """
        The mean sits where the coincidence would be at zero lag.

        Otherwise a coincidence's time drifts with the lag it was found at, and two
        background events a window apart in physical time stop being a window apart in
        the time they are clustered on -- so how much the background clusters would
        depend on how deep in the ladder it came from.
        """
        plain, _ = coincidence_time(
            {"H1": np.array([100.0]), "L1": np.array([100.0])}
        )
        slid, _ = coincidence_time(
            {"H1": np.array([100.0]), "L1": np.array([400.0])},
            offsets_s={"H1": 0.0, "L1": 300.0},
        )
        assert slid[0] == pytest.approx(plain[0])

    def test_empty_coincidence_refused(self):
        """An all-sentinel row has no time; inventing one would place it anywhere."""
        with pytest.raises(ValueError, match="no participating detector"):
            coincidence_time({"H1": np.array([-1.0]), "L1": np.array([-1.0])})


class TestEventGrouping:
    """The wider, cross-pipeline grouping used for catalogue comparison."""

    def test_nearby_times_group(self):
        """The comparison convention: nearby times from two pipelines are one event."""
        times = np.array([100.0, 100.4, 100.8, 140.0, 140.2])
        assert group_events(times).tolist() == [0, 0, 0, 1, 1]

    def test_grouping_single_linkage(self):
        """
        A chain of near-coincident times is one event however far it walks.

        Deliberately not the clustering rule: this answers whether two pipelines mean the
        same event, and there is no statistic shared between them to anchor a peak on.
        """
        times = np.arange(10) * 0.9
        assert group_events(times, window_s=1.0).tolist() == [0] * 10

    def test_empty_groups_to_nothing(self):
        """An empty candidate list is a normal state for an unmatched catalogue."""
        assert group_events(np.array([])).tolist() == []
