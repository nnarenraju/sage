#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_crossmatch.py
Description   : Matching candidates against published catalogues.

Created on 2026-08-20

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later

This is where a search says "we found something nobody published" and where it says "we
missed something everybody published". Both claims are made from the same matching, and
both are wrong in the same ways: a match that fires twice turns one recovery into two, a
match that misses turns a recovery into a discovery, and treating absence outside somebody
else's search as a null result invents discoveries at the edge of their parameter space.
"""

import numpy as np
import pytest

from sage.search.catalogue.record import CatalogueEvent, Conventions, ExternalCatalogue
from sage.search.crossmatch import (
    classify,
    comparison_table,
    coverage_mask,
    match_on_gps,
    overlap_sets,
    union_times,
)


def _catalogue(key, times, names=None, far=None, conventions=None):
    """A published catalogue with the given event times."""
    names = names or [f"{key.upper()}{i}" for i in range(len(times))]
    return ExternalCatalogue(
        key=key,
        events=[
            CatalogueEvent(
                name=name,
                gps=float(time),
                source=key,
                far_per_yr=None if far is None else float(far[i]),
            )
            for i, (time, name) in enumerate(zip(times, names))
        ],
        conventions=conventions or Conventions(significance="far_per_yr"),
    )


class TestMatching:
    """One event, one match."""

    def test_nearest_within_tolerance(self):
        result = match_on_gps(np.array([100.0, 500.0]), np.array([100.3]), 1.0)
        assert result.left_index.tolist() == [0]
        assert np.isclose(result.dt_s[0], -0.3)

    def test_outside_tolerance_does_not_match(self):
        result = match_on_gps(np.array([100.0]), np.array([102.0]), 1.0)
        assert result.left_index.size == 0
        assert result.unmatched_left.tolist() == [0]

    def test_one_to_one(self):
        """
        Two candidates near one published event: only the closer may claim it. Both
        claiming it would report the recovery twice and hide that the other is unmatched.
        """
        result = match_on_gps(np.array([100.0, 100.3]), np.array([100.1]), 1.0)
        assert result.left_index.tolist() == [0]
        assert result.unmatched_left.tolist() == [1]

    def test_assignment_is_order_independent(self):
        """
        Pairs are taken closest-first, so the answer does not depend on the order the
        lists arrive in -- a greedy sweep in index order gives a different answer and
        both look equally reasonable afterwards.
        """
        left = np.array([100.0, 100.3])
        forward = match_on_gps(left, np.array([100.1]), 1.0)
        reverse = match_on_gps(left[::-1], np.array([100.1]), 1.0)
        assert abs(float(forward.dt_s[0])) == abs(float(reverse.dt_s[0]))

    def test_empty_lists(self):
        result = match_on_gps(np.zeros(0), np.array([1.0]), 1.0)
        assert result.left_index.size == 0
        assert result.unmatched_right.tolist() == [0]

    def test_tolerance_must_be_positive(self):
        with pytest.raises(ValueError, match="tolerance_s"):
            match_on_gps(np.array([1.0]), np.array([1.0]), 0.0)

    def test_precision_is_nanoseconds(self):
        """
        Sources quote times to differing precision, and floats of order 1.2e9 carry about
        0.24 us of resolution -- enough for a tolerance to behave differently depending
        on which source was read first.
        """
        base = 1.238e9
        result = match_on_gps(
            np.array([base]), np.array([base + 0.5]), tolerance_s=1.0
        )
        assert result.left_index.tolist() == [0]


class TestClassification:
    """New, known, recovered, missed -- and where the question was never asked."""

    def _setup(self):
        gwtc = _catalogue("gwtc", [100.0, 200.0, 300.0], ["GW1", "GW2", "GW3"])
        candidates = {
            "gps": np.array([100.1, 300.2, 250.0, 9000.0]),
            "mchirp": np.full(4, 25.0),
        }
        return candidates, {"gwtc": gwtc}

    def test_matched_candidates_are_known(self):
        candidates, catalogues = self._setup()
        out = classify(candidates, catalogues, tolerance_s=1.0)
        assert out["known"].tolist() == [True, True, False, False]
        assert out["catalogue_match"][0] == "GW1"

    def test_unmatched_inside_coverage_is_new(self):
        """The claim the search exists to make."""
        candidates, catalogues = self._setup()
        out = classify(candidates, catalogues, tolerance_s=1.0)
        assert out["is_new"][2]

    def test_declared_span_beats_the_event_span(self):
        """
        A source that publishes nothing near the start of a run still searched there, so
        the event span is only a lower bound on coverage. Where a source states its span,
        that is used.
        """
        stated = _catalogue(
            "wide",
            [500.0],
            conventions=Conventions(
                significance="far_per_yr", searched_gps_span=(0.0, 1000.0)
            ),
        )
        implied = _catalogue("narrow", [500.0])
        gps = np.array([100.0])
        assert coverage_mask(stated, gps).tolist() == [True]
        assert coverage_mask(implied, gps).tolist() == [False]

    def test_outside_coverage_is_not_new(self):
        """
        A catalogue that did not search a time says nothing about it. Calling a candidate
        there "new" would be a discovery manufactured from somebody else's scope.
        """
        candidates, catalogues = self._setup()
        out = classify(candidates, catalogues, tolerance_s=1.0)
        assert not out["covered_by_any"][3]
        assert not out["is_new"][3]

    def test_missed_events_are_reported(self):
        """The validation gate: a search that misses published events is not ready."""
        candidates, catalogues = self._setup()
        out = classify(candidates, catalogues, tolerance_s=1.0)
        recovered = out["gwtc_event_recovered"]
        names = out["gwtc_event_name"]
        assert dict(zip(names.tolist(), recovered.tolist())) == {
            "GW1": True,
            "GW2": False,
            "GW3": True,
        }

    def test_mass_coverage_is_respected(self):
        """
        A catalogue that searched a mass range says nothing outside it, so a candidate
        there is not new on its evidence.
        """
        catalogue = _catalogue(
            "narrow",
            [100.0],
            conventions=Conventions(
                significance="far_per_yr",
                searched_mass_range=(20.0, 30.0),
                # Stated, because a one-event catalogue's own span is a single instant
                # and the fallback would report no coverage anywhere.
                searched_gps_span=(0.0, 1000.0),
            ),
        )
        gps = np.array([150.0, 150.0])
        mchirp = np.array([25.0, 60.0])
        covered = coverage_mask(catalogue, gps, mchirp)
        assert covered.tolist() == [True, False]


class TestOverlaps:
    """Counting distinct events across sources."""

    def test_sets_partition_the_union(self):
        times = {"a": np.array([100.0, 200.0]), "b": np.array([100.2, 300.0])}
        sets = overlap_sets(times, tolerance_s=1.0)
        assert sum(len(v) for v in sets.values()) == union_times(times, 1.0).size

    def test_shared_event_is_counted_once(self):
        times = {"a": np.array([100.0]), "b": np.array([100.2])}
        assert union_times(times, 1.0).size == 1
        assert ("a", "b") in overlap_sets(times, 1.0)

    def test_grouping_is_transitive(self):
        """
        A within tolerance of B and B of C groups all three, even where A and C are
        further apart. Two sources quoting a time to different precision are this case,
        and splitting them would report one event as two.
        """
        times = {"a": np.array([100.0]), "b": np.array([100.8]), "c": np.array([101.5])}
        assert union_times(times, 1.0).size == 1

    def test_comparison_keeps_conventions_apart(self):
        """
        Each column carries what its catalogue publishes, under its own conventions. A
        FAR and a p_astro are different quantities and are never placed on one axis.
        """
        gwtc = _catalogue("gwtc", [100.0], far=[0.01])
        other = _catalogue(
            "other",
            [100.2],
            conventions=Conventions(significance="p_astro", pastro_prior="theirs"),
        )
        report = comparison_table(
            {"gps": np.array([100.1])}, {"gwtc": gwtc, "other": other}, 1.0
        )
        assert report["significance_by_source"] == {
            "gwtc": "far_per_yr",
            "other": "p_astro",
        }
        assert "gwtc_far_per_yr" in report["table"]
        assert "other_p_astro" in report["table"]


class TestConventions:
    """What may be compared with what."""

    def test_far_and_pastro_are_not_comparable(self):
        far = Conventions(significance="far_per_yr")
        pastro = Conventions(significance="p_astro", pastro_prior="a")
        assert not far.significance_comparable_to(pastro)

    def test_pastro_under_different_priors_is_not_comparable(self):
        """
        The prior is what turns a likelihood ratio into a probability, so the same event
        scores differently under each and the difference says nothing about the data.
        """
        a = Conventions(significance="p_astro", pastro_prior="a")
        b = Conventions(significance="p_astro", pastro_prior="b")
        assert not a.significance_comparable_to(b)
        assert a.significance_comparable_to(
            Conventions(significance="p_astro", pastro_prior="a")
        )


class TestBBHFilter:
    """The recovery gate is scored against the BBH subset."""

    def test_missing_secondary_mass_is_kept(self):
        """
        Absence of a measurement is not evidence of a light companion. Dropping such
        events would shrink the list the search is scored against, turning a missing
        column into a missed recovery.
        """
        catalogue = ExternalCatalogue(
            key="x",
            events=[
                CatalogueEvent(name="known", gps=1.0, source="x", mass2=25.0),
                CatalogueEvent(name="light", gps=2.0, source="x", mass2=1.4),
                CatalogueEvent(name="unmeasured", gps=3.0, source="x"),
            ],
            conventions=Conventions(),
        )
        kept = {e.name for e in catalogue.filter_bbh(min_secondary_mass=3.0).events}
        assert kept == {"known", "unmeasured"}


class TestRestriction:
    """A catalogue is judged only over the time the search analysed."""

    def _intervals(self):
        return np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float64)

    def test_events_outside_are_dropped(self):
        """
        An event outside the analysed time was never searched for. Reporting it as missed
        makes an O3a search answer for O4 -- measured, before this: 109 O4 events listed
        as missed by an O3a campaign.
        """
        from sage.search.crossmatch import _restrict

        catalogue = _catalogue("x", [150.0, 250.0, 350.0, 9.0e8])
        kept = _restrict(catalogue, self._intervals())
        assert [e.gps for e in kept.events] == [150.0, 350.0]

    def test_span_is_recorded_on_the_conventions(self):
        """
        So that coverage_mask -- which decides whether a *candidate* may be called new --
        uses the analysed bounds rather than falling back to the span of whichever events
        happened to survive.
        """
        from sage.search.crossmatch import _restrict

        kept = _restrict(_catalogue("x", [150.0, 350.0]), self._intervals())
        assert kept.conventions.searched_gps_span == (100.0, 400.0)

    def test_gaps_between_intervals_are_respected(self):
        """
        Analysed time is a union of intervals, not a span. An event in the gap between
        two of them falls in no analysed second.
        """
        from sage.search.crossmatch import _restrict

        kept = _restrict(_catalogue("x", [250.0]), self._intervals())
        assert len(kept) == 0

    def test_no_analysed_time_keeps_nothing(self):
        from sage.search.crossmatch import _restrict

        kept = _restrict(_catalogue("x", [150.0]), np.zeros((0, 2)))
        assert len(kept) == 0
        assert kept.conventions.searched_gps_span is None


class TestOfflineCache:
    """An analysis whose inputs can change under it between runs is not reproducible."""

    def test_uncached_url_is_refused_offline(self, tmp_path):
        from sage.search.catalogue.cache import CatalogueCache

        cache = CatalogueCache(tmp_path / "c", offline_only=True)
        with pytest.raises(LookupError, match="offline"):
            cache.fetch("https://example.invalid/never-fetched")

    def test_entries_survive_a_new_instance(self, tmp_path):
        """
        The entry table is persisted on every store, not only at freeze(). Held in memory
        alone, the first run populates the cache and the next one silently goes to the
        network -- so "fetched once" would be untrue after the first run.
        """
        from sage.search.catalogue.cache import CatalogueCache

        first = CatalogueCache(tmp_path / "c")
        entry = first.put("https://example.invalid/x", b"payload")
        second = CatalogueCache(tmp_path / "c", offline_only=True)
        assert second.fetch("https://example.invalid/x").sha256 == entry.sha256

    def test_tampering_is_caught_by_rehashing(self, tmp_path):
        """
        A truncated or rewritten file keeps a plausible size, and the failure it produces
        -- a catalogue short of its last events -- reads as a real difference in the
        comparison rather than as corruption.
        """
        from sage.search.catalogue.cache import CatalogueCache

        cache = CatalogueCache(tmp_path / "c")
        entry = cache.put("https://example.invalid/x", b"payload")
        manifest = cache.freeze(tmp_path / "c" / "catalogue_cache.json")
        assert all(cache.verify(manifest).values())
        entry.path.write_bytes(b"tampered")
        assert not any(cache.verify(manifest).values())


class TestMatchTime:
    """
    Which of a candidate's two times a catalogue is joined on.

    A published event's GPS is a *merger* time. A candidate carries both the analysis
    window's start (``gps``) and the decoded coalescence time (``tc_gps``), and on the O3a
    smoke campaign those differ by 13.05 s -- twenty times the 1.0 s tolerance. Joining on
    the window start reported every recovered event as a new discovery.
    """

    def _pair(self, offset=13.05):
        """One candidate whose window starts well before the merger it contains."""
        return {
            "gps": np.array([1238782687.24]),
            "tc_gps": np.array([1238782687.24 + offset]),
            "mchirp": np.array([25.0]),
        }

    def test_joined_on_the_merger_time(self):
        candidates = self._pair()
        catalogue = _catalogue(
            "gwtc", [float(candidates["tc_gps"][0])], ["GW190408_181802"]
        )
        out = classify(candidates, {"gwtc": catalogue}, tolerance_s=1.0)
        assert out["known"].tolist() == [True]
        assert out["catalogue_match"][0] == "GW190408_181802"

    def test_window_start_does_not_match(self):
        """The negative control: the same event at the window start is 13 s away."""
        candidates = self._pair()
        catalogue = _catalogue("gwtc", [float(candidates["gps"][0])], ["GW190408_181802"])
        out = classify(candidates, {"gwtc": catalogue}, tolerance_s=1.0)
        assert out["known"].tolist() == [False]

    def test_falls_back_to_the_window_start(self):
        """
        A campaign whose engine carried no decoder estimated no coalescence time.

        It cannot place a candidate to better than a window, and matching on the window
        start is the honest thing left to do -- but it must not silently raise.
        """
        from sage.search.crossmatch import merger_times

        columns = {"gps": np.array([100.0, 200.0])}
        assert merger_times(columns).tolist() == [100.0, 200.0]

    def test_prefers_tc_when_both_are_present(self):
        from sage.search.crossmatch import merger_times

        columns = {"gps": np.array([100.0]), "tc_gps": np.array([113.05])}
        assert merger_times(columns).tolist() == [113.05]
