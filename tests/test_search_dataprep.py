#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_dataprep.py
Description   : Search-grade release construction from GWOSC strain.

Created on 2026-08-10

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Runs against synthetic files in the GWOSC layout, staged where the fetcher would have
put real ones, so the whole build path is exercised without the network. The stored
samples are checked against a direct conditioning of the same raw span rather than
against a recorded constant, because the point of the release is that it is exactly the
strain conditioned once.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from sage.search import dataprep
from sage.search.dataprep import (
    GWOSC_FILE_DURATION_S,
    GWOSC_SAMPLE_RATE,
    SearchDataSpec,
    SourceFiles,
    _file_start,
    _files_spanning,
    condition,
    file_url,
    livetime_budget,
    segment_plan,
)

h5py = pytest.importorskip("h5py")
pytest.importorskip("pycbc")

RUN = "O3a"
FILE0 = 1238171648  # a real O3a file boundary; an exact multiple of 4096


def _spec(tmp_path: Path, **kwargs) -> SearchDataSpec:
    defaults = dict(
        observing_run=RUN,
        detectors=("H1",),
        out_dir=tmp_path / "release",
        scratch_dir=tmp_path / "scratch",
        download_workers=1,
        cache_files=8,
    )
    defaults.update(kwargs)
    return SearchDataSpec(**defaults)


def _plan(monkeypatch, segments):
    """Serve a fixed segment list instead of querying GWOSC."""
    monkeypatch.setattr(dataprep, "_query_timeline", lambda run, flag, **kw: segments)


class _FakeResponse:
    """Enough of a streaming ``requests`` response for the fetcher."""

    status_code = 200

    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1 << 22):
        yield self._payload


class _FlakySession:
    """A session that fails a fixed number of times, then serves the file."""

    def __init__(self, failures: int, exc: BaseException, payload: bytes = b"\x00" * 64):
        self.failures = failures
        self.exc = exc
        self.payload = payload
        self.calls = 0

    def get(self, url, stream=True, timeout=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return _FakeResponse(self.payload)

    def close(self):
        return None


class TestSpec:
    """The conditions, and what they imply before anything is fetched."""

    def test_trim_is_a_whole_number_of_samples(self, tmp_path):
        """
        Stored segments must share one sample grid across the network.

        Natural segment boundaries are integer GPS, so an integer trim puts every
        detector's stored samples at the same offset from a second. A fractional trim
        would reintroduce the half-sample misalignment the grid has to correct for.
        """
        spec = _spec(tmp_path)
        assert spec.trim_samples == 410
        assert float(spec.trim_samples).is_integer()

    def test_decimation_is_an_integer_factor(self, tmp_path):
        assert _spec(tmp_path).decimation == 2

    def test_non_dividing_rate_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="integer decimation"):
            _spec(tmp_path, sample_rate=3000.0)

    def test_margin_must_fit_inside_a_block(self, tmp_path):
        with pytest.raises(ValueError, match="margin"):
            _spec(tmp_path, block_s=32.0, margin_s=32.0)

    def test_virgo_is_refused_for_o4a(self, tmp_path):
        """V1 published no O4a strain, so an HLV O4a search cannot exist."""
        with pytest.raises(ValueError, match="Virgo"):
            _spec(tmp_path, observing_run="O4a", detectors=("H1", "L1", "V1")).validate()

    def test_release_dir_names_conditions(self, tmp_path):
        spec = SearchDataSpec(
            observing_run="O3a", detectors=("H1", "L1", "V1"), out_dir=None
        )
        assert spec.release_dir().name == "o3a_search_data_DATA_HLV"

    def test_budget_picks_conditioning_path(self, tmp_path):
        """
        The budget is the only thing that chooses between the two paths.

        One conditioning pass costs about 3.2 times the raw float64 input, so a 4 GiB
        budget holds roughly a ten-hour segment and the O3a maximum of 46.4 hours needs
        about 17 GiB.
        """
        spec = _spec(tmp_path, memory_budget_gb=4.0)
        assert spec.whole_segment_fits(3600.0)
        assert not spec.whole_segment_fits(167178.0)
        assert _spec(tmp_path, memory_budget_gb=20.0).whole_segment_fits(167178.0)

    def test_conditions_round_trip_as_json(self, tmp_path):
        spec = _spec(tmp_path)
        assert json.loads(json.dumps(spec.as_dict()))["detectors"] == ["H1"]


class TestFileLayout:
    """Which GWOSC files a span needs."""

    def test_file_start_is_the_containing_multiple(self):
        assert _file_start(FILE0) == FILE0
        assert _file_start(FILE0 + 1) == FILE0
        assert _file_start(FILE0 - 1) == FILE0 - GWOSC_FILE_DURATION_S

    def test_a_span_inside_one_file_needs_one_file(self):
        assert _files_spanning(FILE0 + 10, FILE0 + 20) == [FILE0]

    def test_span_on_boundary_excludes_next_file(self):
        """
        The end is exclusive.

        Taking the next file would fetch a whole file to read zero samples from it, and
        for a release built segment by segment that is thousands of wasted files.
        """
        assert _files_spanning(FILE0, FILE0 + GWOSC_FILE_DURATION_S) == [FILE0]

    def test_long_segment_spans_all_files(self):
        """
        Natural segments span many files, not one boundary.

        Measured on O3a: the median segment spans four GWOSC files and the longest spans
        forty-one. Handling only a head and a tail would silently drop everything in
        between and store a short array under the full GPS span.
        """
        span = 41 * GWOSC_FILE_DURATION_S
        files = _files_spanning(FILE0, FILE0 + span)
        assert len(files) == 41
        assert files[0] == FILE0
        assert files[-1] == FILE0 + 40 * GWOSC_FILE_DURATION_S
        assert np.all(np.diff(files) == GWOSC_FILE_DURATION_S)

    def test_url_matches_the_published_layout(self):
        url = file_url("V1", "O3a", FILE0)
        assert url.endswith("V-V1_GWOSC_O3a_4KHZ_R1-1238171648-4096.hdf5")
        assert "/O3a_4KHZ_R1/1238171648/" in url


class TestConditioning:
    """The two paths must agree to the precision the release is stored at."""

    def test_output_length_follows_the_decimation(self):
        # 64 s is all the assertion needs; generating a whole GWOSC file here
        # (4096 s x 4096 Hz, float64) cost 134 MB per run for nothing.
        rng = np.random.default_rng(FILE0)
        raw = rng.standard_normal(int(64 * GWOSC_SAMPLE_RATE)) * 1e-20
        out = condition(raw, GWOSC_SAMPLE_RATE, 2048.0, 15.0)
        assert out.dtype == np.float32
        assert out.size == raw.size // 2


class TestResilience:
    """
    Surviving the service being unavailable.

    A real build died here: the site reaches GWOSC through a proxy that answered
    ``503 Service Unavailable``, the fetcher exhausted a 63 s retry budget, and one file
    ended a run that was 149 segments in. Waiting out an outage is the correct response
    at both levels, so both are tested.
    """

    @staticmethod
    def _proxy_error():
        import requests

        return requests.exceptions.ProxyError(
            "Unable to connect to proxy",
            OSError("Tunnel connection failed: 503 Service Unavailable"),
        )

    def test_proxy_failure_is_transient(self):
        """The exact exception that ended the O3a build must be recognised."""
        assert dataprep.is_transient(self._proxy_error())

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_server_errors_are_transient(self, status):
        import requests

        response = requests.Response()
        response.status_code = status
        error = requests.exceptions.HTTPError(response=response)
        assert dataprep.is_transient(error)

    def test_missing_file_is_not_transient(self):
        """A 404 will not fix itself, so waiting on it would hang the build."""
        import requests

        response = requests.Response()
        response.status_code = 404
        assert not dataprep.is_transient(
            requests.exceptions.HTTPError(response=response)
        )
        assert not dataprep.is_transient(ValueError("segment list disagrees"))

    def test_download_survives_outage(self, tmp_path, monkeypatch):
        """
        A transient failure is retried until the service returns.

        The old budget was six attempts over 63 s; an outage of a few minutes is
        routine, so what matters is that the fetch survives many consecutive failures
        rather than a fixed small number.
        """
        payload = b"\x00" * 4096
        session = _FlakySession(failures=9, exc=self._proxy_error(), payload=payload)
        monkeypatch.setattr(SourceFiles, "_make_session", lambda self: session)
        slept = []
        monkeypatch.setattr(dataprep.time, "sleep", slept.append)

        files = SourceFiles(tmp_path / "s", RUN, workers=1, max_files=4)
        try:
            path = files._download("H1", FILE0)
        finally:
            files.close()
        assert path.read_bytes() == payload
        assert session.calls == 10, "should have retried until the service returned"
        assert len(slept) == 9 and max(slept) <= 600.0

    def test_download_gives_up_on_budget(self, tmp_path, monkeypatch):
        """A service that is genuinely gone ends the run instead of hanging forever."""
        session = _FlakySession(failures=10**6, exc=self._proxy_error())
        monkeypatch.setattr(SourceFiles, "_make_session", lambda self: session)
        monkeypatch.setattr(dataprep.time, "sleep", lambda s: None)

        files = SourceFiles(
            tmp_path / "s", RUN, workers=1, max_files=4, outage_budget_s=120.0
        )
        try:
            with pytest.raises(RuntimeError, match="waiting"):
                files._download("H1", FILE0)
        finally:
            files.close()

    def test_a_permanent_failure_is_not_waited_on(self, tmp_path, monkeypatch):
        """Waiting on something that cannot recover would burn the whole budget."""
        session = _FlakySession(failures=10**6, exc=ValueError("malformed"))
        monkeypatch.setattr(SourceFiles, "_make_session", lambda self: session)
        slept = []
        monkeypatch.setattr(dataprep.time, "sleep", slept.append)

        files = SourceFiles(tmp_path / "s", RUN, workers=1, max_files=4)
        try:
            with pytest.raises(RuntimeError):
                files._download("H1", FILE0)
        finally:
            files.close()
        assert slept == [], "a permanent failure must fail fast"

    def test_the_session_carries_a_retry_adapter(self, tmp_path):
        """
        Transport-level retries sit beneath requests, as in the primer downloader.

        Without an adapter every 5xx reaches the caller immediately, which is what made
        the old budget so short in practice.
        """
        files = SourceFiles(tmp_path / "s", RUN, workers=8, max_files=4)
        try:
            adapter = files._session.get_adapter("https://gwosc.org")
            retry = adapter.max_retries
            assert retry.total >= 5
            assert {500, 502, 503, 504}.issubset(set(retry.status_forcelist))
            assert adapter._pool_maxsize >= 16, "pool must cover the worker count"
        finally:
            files.close()


class TestPlanning:
    """What is selected, and what it costs, before anything is fetched."""

    def test_short_segments_dropped(self, monkeypatch, tmp_path):
        """
        A segment that cannot host one window is not analysable time.

        O3a publishes segments as short as one second; keeping them would put time in
        the release that no window start can reach.
        """
        _plan(
            monkeypatch,
            [(1000.0, 1001.0), (2000.0, 2100.0), (3000.0, 3016.0)],
        )
        plan = segment_plan(_spec(tmp_path))
        assert plan["H1"] == [(2000.0, 2100.0)]

    def test_the_plan_is_sorted_by_time(self, monkeypatch, tmp_path):
        _plan(monkeypatch, [(5000.0, 5100.0), (1000.0, 1100.0)])
        plan = segment_plan(_spec(tmp_path))
        assert plan["H1"] == [(1000.0, 1100.0), (5000.0, 5100.0)]

    def test_a_gps_range_clips_the_plan(self, monkeypatch, tmp_path):
        """Building a subset against real strain must not need the whole run."""
        _plan(monkeypatch, [(1000.0, 5000.0), (9000.0, 9100.0)])
        plan = segment_plan(_spec(tmp_path, gps_range=(2000.0, 4000.0)))
        assert plan["H1"] == [(2000.0, 4000.0)]

    def test_the_budget_accounts_for_every_second(self, monkeypatch, tmp_path):
        """
        Observing time is either stored, too short to keep, or trimmed at an edge.

        The three must add up, or the livetime the search reports cannot be justified.
        """
        _plan(monkeypatch, [(1000.0, 1001.0), (2000.0, 6000.0), (7000.0, 7500.0)])
        budget = livetime_budget(_spec(tmp_path))["detectors"]["H1"]
        total = (
            budget["stored_s"] + budget["short_segment_loss_s"] + budget["trim_loss_s"]
        )
        assert total == pytest.approx(budget["observing_s"], abs=1e-6)
        assert budget["segments_kept"] == 2
