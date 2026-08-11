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
    known_event_coverage,
    livetime_budget,
    load_release_segments,
    prepare,
    read_segment,
    segment_plan,
    sidecar_path,
    strain_path,
    verify,
)

h5py = pytest.importorskip("h5py")
pytest.importorskip("pycbc")

RUN = "O3a"
FILE0 = 1238171648  # a real O3a file boundary; an exact multiple of 4096


def _raw(file_start: int) -> np.ndarray:
    """Deterministic strain-scale samples for one synthetic GWOSC file."""
    rng = np.random.default_rng(int(file_start))
    n = int(GWOSC_FILE_DURATION_S * GWOSC_SAMPLE_RATE)
    return rng.standard_normal(n) * 1e-20


def _stage(base: Path, detector: str, file_starts, nan_span=None) -> None:
    """
    Publish synthetic GWOSC files to a fake remote.

    Not written straight into the staging area: staged files are a cache that the
    fetcher is free to evict and re-fetch, so a fixture that put the only copy there
    would break the moment eviction worked correctly. ``_serve`` supplies the fetch.
    """
    root = base / "remote"
    root.mkdir(parents=True, exist_ok=True)
    for file_start in file_starts:
        data = _raw(file_start)
        if nan_span is not None:
            lo, hi = nan_span
            i0 = int((lo - file_start) * GWOSC_SAMPLE_RATE)
            i1 = int((hi - file_start) * GWOSC_SAMPLE_RATE)
            if i1 > 0 and i0 < data.size:
                data[max(0, i0) : min(data.size, i1)] = np.nan
        with h5py.File(root / f"{detector}-{file_start}.hdf5", "w") as handle:
            handle.create_dataset("strain/Strain", data=data)
            handle.create_dataset("meta/Duration", data=float(GWOSC_FILE_DURATION_S))
            handle.create_dataset("meta/GPSstart", data=float(file_start))


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


def _serve(monkeypatch, base: Path):
    """
    Fetch from the fake remote instead of the network.

    Stands in for the HTTP transfer only; staging, eviction and re-fetching all run as
    they do in production, so a file evicted from the cache is fetched again rather than
    being lost.
    """
    import shutil

    def fake_download(self, detector, file_start, retries=6):
        target = self._path(detector, int(file_start))
        if not target.exists():
            source = base / "remote" / f"{detector}-{int(file_start)}.hdf5"
            if not source.exists():
                raise RuntimeError(f"GWOSC does not publish {source.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return target

    monkeypatch.setattr(SourceFiles, "_download", fake_download)


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

    def test_release_directory_states_the_conditions(self, tmp_path):
        spec = SearchDataSpec(
            observing_run="O3a", detectors=("H1", "L1", "V1"), out_dir=None
        )
        assert spec.release_dir().name == "o3a_search_data_DATA_HLV"

    def test_memory_budget_decides_the_conditioning_path(self, tmp_path):
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

    def test_a_span_ending_on_a_boundary_does_not_take_the_next_file(self):
        """
        The end is exclusive.

        Taking the next file would fetch a whole file to read zero samples from it, and
        for a release built segment by segment that is thousands of wasted files.
        """
        assert _files_spanning(FILE0, FILE0 + GWOSC_FILE_DURATION_S) == [FILE0]

    def test_a_long_segment_needs_every_file_it_crosses(self):
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


class TestSourceFiles:
    """Assembling raw strain out of staged files."""

    def test_reads_a_span_crossing_several_files(self, tmp_path, monkeypatch):
        starts = [FILE0 + k * GWOSC_FILE_DURATION_S for k in range(4)]
        _stage(tmp_path, "H1", starts)
        _serve(monkeypatch, tmp_path)
        files = SourceFiles(tmp_path / "scratch", RUN, workers=1, max_files=8)
        try:
            lo = FILE0 + 100
            hi = FILE0 + 3 * GWOSC_FILE_DURATION_S + 50
            got = files.read("H1", lo, hi)
            want = np.concatenate([_raw(s) for s in starts])[
                int(100 * GWOSC_SAMPLE_RATE) :
                int((3 * GWOSC_FILE_DURATION_S + 50) * GWOSC_SAMPLE_RATE)
            ]
            assert got.size == int((hi - lo) * GWOSC_SAMPLE_RATE)
            assert np.array_equal(got, want)
        finally:
            files.close()

    def test_unset_samples_are_refused(self, tmp_path, monkeypatch):
        """
        GWOSC pads non-observing time with NaN.

        A NaN inside a span means the segment list and the strain disagree; one NaN
        propagates through the whole conditioning pass, so this must fail loudly rather
        than store a segment of NaN.
        """
        _stage(tmp_path, "H1", [FILE0], nan_span=(FILE0 + 10, FILE0 + 20))
        _serve(monkeypatch, tmp_path)
        files = SourceFiles(tmp_path / "scratch", RUN, workers=1, max_files=4)
        try:
            with pytest.raises(ValueError, match="unset samples"):
                files.read("H1", FILE0, FILE0 + 100)
        finally:
            files.close()

    def test_a_resumed_run_adopts_what_a_previous_one_staged(self, tmp_path, monkeypatch):
        """
        Files left by an earlier run must come under the cap, not sit on top of it.

        The eviction list is built as files are acquired, so without adopting them a
        resumed build holds every previously-staged file for its whole life. The staging
        area shares a storage quota with the release, so that overrun is charged against
        the very thing being built. Measured on the real O3a build: a resumed run sat at
        14.2 GiB against a 5.8 GiB cap until the leftovers were cleared by hand.
        """
        starts = [FILE0 + k * GWOSC_FILE_DURATION_S for k in range(10)]
        _stage(tmp_path, "H1", starts)
        _serve(monkeypatch, tmp_path)

        # A previous run's staging area: fetched, then abandoned when it ended.
        first = SourceFiles(tmp_path / "scratch", RUN, workers=1, max_files=10)
        for start in starts:
            first.acquire("H1", start)
        first.close()
        staged = tmp_path / "scratch" / RUN
        assert len(list(staged.glob("H1-*.hdf5"))) == 10

        resumed = SourceFiles(tmp_path / "scratch", RUN, workers=1, max_files=3)
        try:
            resident = list(staged.glob("H1-*.hdf5"))
            assert len(resident) <= 3, f"{len(resident)} left staged, cap is 3"
        finally:
            resumed.close()

    def test_staging_area_stays_bounded(self, tmp_path, monkeypatch):
        """Files are evicted once read, so the staging area does not grow with the run."""
        starts = [FILE0 + k * GWOSC_FILE_DURATION_S for k in range(6)]
        _stage(tmp_path, "H1", starts)
        _serve(monkeypatch, tmp_path)
        files = SourceFiles(tmp_path / "scratch", RUN, workers=1, max_files=2)
        try:
            for start in starts:
                files.read("H1", start, start + 10)
            resident = list((tmp_path / "scratch" / RUN).glob("H1-*.hdf5"))
            assert len(resident) <= 2
        finally:
            files.close()


class TestConditioning:
    """The two paths must agree to the precision the release is stored at."""

    def test_output_length_follows_the_decimation(self):
        raw = _raw(FILE0)[: int(64 * GWOSC_SAMPLE_RATE)]
        out = condition(raw, GWOSC_SAMPLE_RATE, 2048.0, 15.0)
        assert out.dtype == np.float32
        assert out.size == raw.size // 2

    def test_blocked_matches_single_pass_at_float32_precision(self, tmp_path, monkeypatch):
        """
        Blocking is a memory strategy, not a different computation.

        Both filters are zero-phase IIR, so a block conditioned with margins differs
        from the same stretch conditioned whole only by a transient that has decayed
        below the stored precision. Measured on real O3a strain the residual floors at
        about 1.3e-7 of the segment's rms and does not fall with a larger margin, which
        is float32 rounding on a sample of typical size. The bound is on the absolute
        difference, not on ulp distance: a difference of one rounding step lands on a
        near-zero sample often enough that ulp distance reaches the thousands while the
        strain itself agrees to seven digits.
        """
        starts = [FILE0 + k * GWOSC_FILE_DURATION_S for k in range(2)]
        _stage(tmp_path, "H1", starts)
        _serve(monkeypatch, tmp_path)
        files = SourceFiles(tmp_path / "scratch", RUN, workers=1, max_files=4)
        lo, hi = float(FILE0), float(FILE0 + 6000)
        try:
            spec = _spec(tmp_path, block_s=512.0, margin_s=32.0)
            whole = np.concatenate(
                list(dataprep._iter_conditioned(files, "H1", lo, hi, spec, True))
            )
            blocked = np.concatenate(
                list(dataprep._iter_conditioned(files, "H1", lo, hi, spec, False))
            )
        finally:
            files.close()
        assert whole.shape == blocked.shape
        rms = float(np.sqrt(np.mean(whole.astype(np.float64) ** 2)))
        worst = float(np.max(np.abs(whole.astype(np.float64) - blocked)))
        assert worst / rms < 5e-7, f"{worst / rms:.3e} exceeds float32 rounding"


class TestBuild:
    """End to end, against synthetic files staged where real ones would be."""

    @pytest.fixture
    def built(self, tmp_path, monkeypatch):
        starts = [FILE0 + k * GWOSC_FILE_DURATION_S for k in range(4)]
        _stage(tmp_path, "H1", starts)
        _serve(monkeypatch, tmp_path)
        segments = [
            (float(FILE0 + 100), float(FILE0 + 2000)),
            (float(FILE0 + 5000), float(FILE0 + 3 * GWOSC_FILE_DURATION_S + 500)),
        ]
        _plan(monkeypatch, segments)
        spec = _spec(tmp_path, memory_budget_gb=64.0)
        prepare(spec, progress=False)
        return spec, segments

    def test_every_planned_segment_is_stored(self, built):
        spec, segments = built
        report = verify(spec.release_dir(), spec)
        assert report["ok"], report
        assert report["detectors"]["H1"]["segments"] == len(segments)

    def test_stored_samples_are_the_conditioned_strain(self, built, tmp_path):
        """
        The release is the strain conditioned once, with the trim removed.

        Checked against a direct conditioning of the same raw span rather than a stored
        constant, so a change in either filter shows up here rather than downstream.
        """
        spec, segments = built
        from pycbc import DYN_RANGE_FAC

        gps_start, gps_end = segments[0]
        raw = np.concatenate([_raw(s) for s in _files_spanning(gps_start, gps_end)])
        base = _files_spanning(gps_start, gps_end)[0]
        i0 = int(round((gps_start - base) * GWOSC_SAMPLE_RATE))
        i1 = int(round((gps_end - base) * GWOSC_SAMPLE_RATE))
        want = condition(raw[i0:i1], GWOSC_SAMPLE_RATE, spec.sample_rate, 15.0)
        want = (want * DYN_RANGE_FAC).astype(np.float32)
        trim = spec.trim_samples
        got = read_segment(spec.release_dir(), "H1", RUN, 0)
        assert np.array_equal(got, want[trim : want.size - trim])

    def test_segments_are_gps_ordered_and_disjoint(self, built):
        """
        Unlike the training releases, index order is time order and nothing overlaps.

        The training sidecars are in parallel-completion order with 15.6 s of duplicated
        strain at every boundary; a search that stored time twice would count it twice
        in the background.
        """
        spec, _ = built
        records = load_release_segments(spec.release_dir(), "H1", RUN)
        assert [r.segment_index for r in records] == sorted(
            range(len(records)), key=lambda i: records[i].gps_start
        )
        for earlier, later in zip(records, records[1:]):
            assert earlier.gps_end <= later.gps_start

    def test_sample_index_is_contiguous_across_the_release(self, built):
        spec, _ = built
        records = load_release_segments(spec.release_dir(), "H1", RUN)
        cursor = 0
        for record in records:
            assert record.sample_start_idx == cursor
            cursor += record.nsamples

    def test_the_trim_is_reflected_in_the_stored_times(self, built):
        spec, segments = built
        records = load_release_segments(spec.release_dir(), "H1", RUN)
        offset = spec.trim_samples / spec.sample_rate
        for record, (gps_start, gps_end) in zip(records, segments):
            assert record.gps_start == pytest.approx(gps_start + offset)
            assert record.gps_end == pytest.approx(gps_end - offset)
            assert record.nsamples == int(
                round((gps_end - gps_start) * spec.sample_rate)
            ) - 2 * spec.trim_samples

    def test_checksums_match_what_was_written(self, built):
        spec, _ = built
        report = verify(spec.release_dir(), spec, checksums=True)
        assert report["detectors"]["H1"]["checksum_failures"] == []

    def test_the_master_file_links_every_detector(self, built):
        spec, _ = built
        master = spec.release_dir() / f"data_{RUN}.h5"
        with h5py.File(master, "r") as handle:
            assert handle["H1/segments/000000"].shape[0] > 0

    def test_the_release_records_its_own_conditions(self, built):
        spec, _ = built
        stored = json.loads((spec.release_dir() / "spec.json").read_text())
        assert stored["dq_flag"] == "DATA"
        assert stored["trim_samples"] == spec.trim_samples

    def test_a_large_and_a_small_machine_agree(self, tmp_path, monkeypatch):
        """
        The memory budget must not change the release, only how it is built.

        A budget too small for the segment forces the blocked path; the result has to
        match the single-pass release to the stored precision, or a small machine would
        produce a different dataset from a large one.
        """
        from pycbc import DYN_RANGE_FAC  # noqa: F401 - imported for the same env as build

        starts = [FILE0 + k * GWOSC_FILE_DURATION_S for k in range(2)]
        _stage(tmp_path, "H1", starts)
        _serve(monkeypatch, tmp_path)
        segments = [(float(FILE0), float(FILE0 + 6000))]
        _plan(monkeypatch, segments)

        big = _spec(tmp_path, out_dir=tmp_path / "big", memory_budget_gb=64.0)
        small = _spec(
            tmp_path,
            out_dir=tmp_path / "small",
            memory_budget_gb=0.05,
            block_s=512.0,
            margin_s=32.0,
        )
        prepare(big, progress=False)
        prepare(small, progress=False)

        a = read_segment(big.release_dir(), "H1", RUN, 0).astype(np.float64)
        b = read_segment(small.release_dir(), "H1", RUN, 0).astype(np.float64)
        assert a.shape == b.shape
        rms = float(np.sqrt(np.mean(a**2)))
        assert float(np.max(np.abs(a - b))) / rms < 5e-7

        records = load_release_segments(small.release_dir(), "H1", RUN)
        stored = json.loads(
            sidecar_path(small.release_dir(), "H1", RUN).read_text()
        )
        assert stored[0]["conditioning"] == "blocked"
        assert len(records) == 1

    def test_an_interrupted_build_resumes_from_the_index(self, built, tmp_path):
        """
        Strain is written before its index entry, so a crash costs one segment at most.

        A resumed build must discard the half-written dataset rather than treat it as
        data, and must continue at the same sample cursor.
        """
        spec, segments = built
        target = strain_path(spec.release_dir(), "H1", RUN)
        before = load_release_segments(spec.release_dir(), "H1", RUN)

        # Drop the last index entry, leaving its strain behind as an interrupted build.
        with h5py.File(target, "a") as handle:
            records = dataprep._existing_index(handle)
            dataprep._rewrite_index(handle, records[:-1])
        prepare(spec, progress=False)

        after = load_release_segments(spec.release_dir(), "H1", RUN)
        assert len(after) == len(before)
        assert [r.sample_start_idx for r in after] == [r.sample_start_idx for r in before]
        assert verify(spec.release_dir(), spec, checksums=True)["ok"]

    def test_a_changed_segment_list_stops_a_resume(self, built, monkeypatch):
        """
        Resuming continues by position, so the plan must not have moved underneath it.

        A revised flag would otherwise shift every remaining segment and produce a
        release whose sidecar times do not describe its samples.
        """
        spec, segments = built
        shifted = [(a + 64.0, b + 64.0) for a, b in segments]
        _plan(monkeypatch, shifted)
        with pytest.raises(ValueError, match="segment list has changed"):
            prepare(spec, progress=False)


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

    def test_download_waits_out_an_outage_and_succeeds(self, tmp_path, monkeypatch):
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

    def test_download_gives_up_once_the_budget_is_spent(self, tmp_path, monkeypatch):
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

    def test_a_segment_survives_an_outage_mid_write(self, tmp_path, monkeypatch):
        """
        The layer above the fetcher: a failed segment is retried, not abandoned.

        Segments are written in GPS order with a contiguous sample index, so one cannot
        be skipped and filled in later without reordering the release.
        """
        calls = {"n": 0}
        real = dataprep._write_segment

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise TestResilience._proxy_error()
            return real(*args, **kwargs)

        monkeypatch.setattr(dataprep, "_write_segment", flaky)
        monkeypatch.setattr(dataprep.time, "sleep", lambda s: None)

        starts = [FILE0 + k * GWOSC_FILE_DURATION_S for k in range(2)]
        _stage(tmp_path, "H1", starts)
        _serve(monkeypatch, tmp_path)
        segments = [(float(FILE0), float(FILE0 + 3000))]
        _plan(monkeypatch, segments)
        spec = _spec(tmp_path, memory_budget_gb=64.0)
        prepare(spec, progress=False)

        assert calls["n"] == 3, "two failures then success"
        records = load_release_segments(spec.release_dir(), "H1", RUN)
        assert len(records) == 1
        assert verify(spec.release_dir(), spec, checksums=True)["ok"]


class TestPlanning:
    """What is selected, and what it costs, before anything is fetched."""

    def test_segments_too_short_to_analyse_are_dropped(self, monkeypatch, tmp_path):
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
