#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : dataprep.py
Description   : Prepare search-grade strain for an observing run.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Training noise and search strain are prepared for opposite purposes, and a release built
for one is not usable for the other.

Training wants clean noise: known signals are cut out so the network never learns them as
background, and small losses of time are irrelevant because windows are drawn at random.
A search wants the opposite: every second the detectors were observing, including the
times where signals are known to be, because those are what a search must recover and
what its recovery of known events is judged on. Time lost systematically at a repeating
boundary matters, because it biases livetime and therefore every rate.

The existing releases under ``/work/nagarajan/data_release*`` were built for training.
They exclude known events by design and cut the timeline into fixed 512 s chunks that
overlap by slightly less than one analysis window, leaving a band at every boundary that
can host no window start.

This module builds a separate dataset under its own directory. Nothing existing is
modified or reprocessed: the training datasets remain valid for what they were built for,
and a search reads its own.

Segmentation
------------
The release keeps the detector's natural on/off structure exactly as GWOSC publishes it.
Segments are not cut into chunks and consecutive segments do not overlap, because both
devices exist to serve training and neither helps a search: chunking exists so a segment
can carry its own whitening spectrum, and overlap exists so the random-window sampler
loses nothing at a boundary. A search whitens from fiducial spectra and walks the lattice
in order, so it wants the longest contiguous stretches available and no time stored twice.

The only processing applied is the resample to the analysis rate and the high pass that
must accompany it. Segment boundaries are therefore real gaps in observation rather than
an artefact of the release, and every boundary band lost to the window length is a
genuine consequence of the detector being off.

Memory
------
A natural segment can be long -- the O3a maximum is 46.4 hours -- so conditioning one in
a single pass needs about 3.2 times its raw size in memory, which is 17 GiB for that
segment. That is available on a large node and not on a small one, so the conditioning
follows a memory budget: a segment is conditioned in one pass when it fits, and otherwise
in overlapping blocks whose margins are discarded. Blocked conditioning agrees with
single-pass conditioning to one unit in the last place of the stored float32, measured on
real O3a strain, and the mode used is recorded per segment.

Peak memory is therefore bounded by the budget whatever the segment length, and a larger
budget is used when it is given rather than being an assumption.

See :mod:`sage.search.dqflags` for the flag policy and
:mod:`sage.diagnostics.diagnose_search_data` for an audit of an existing release.
"""

import hashlib
import json
import math
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

Interval = Tuple[float, float]

#: GWOSC publishes strain in files of this duration, starting at exact multiples of it.
GWOSC_FILE_DURATION_S = 4096

#: Sample rate of the ``*_4KHZ_R1`` GWOSC datasets, asserted on every file read.
GWOSC_SAMPLE_RATE = 4096.0

#: Peak resident memory of one conditioning pass, as a multiple of the raw float64 input.
#: Measured on this stack over inputs from 2,000 s to 20,000 s; the ratio falls towards
#: this value as the input grows and the interpreter baseline stops dominating.
CONDITIONING_MEMORY_FACTOR = 3.2

_TIMELINE_URL = (
    "https://gwosc.org/timeline/segments/json/{run}/{flag}/{start}/{duration}/"
)


# ----------------------------------------------------------------------------
# Specification
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchDataSpec:
    """
    Conditions for a search-grade release.

    The defaults describe what a search needs; the resource fields describe what the
    machine building it can afford, and change nothing about the result except which
    conditioning path is taken.
    """

    observing_run: str
    detectors: Tuple[str, ...] = ("H1", "L1", "V1")
    dq_flag: str = "DATA"
    gps_range: Optional[Interval] = None
    out_dir: Optional[Path] = None
    scratch_dir: Optional[Path] = None

    sample_rate: float = 2048.0
    low_frequency_cutoff: float = 15.0
    trim_s: float = 0.2
    window_s: float = 16.0
    minimum_segment_s: Optional[float] = None

    # Resources. None of these alter the selection or the geometry.
    memory_budget_gb: float = 8.0
    block_s: float = 2048.0
    margin_s: float = 32.0
    download_workers: int = 4
    cache_files: int = 24
    min_rate_mb_s: float = 2.0
    stall_grace_s: float = 20.0
    outage_budget_s: float = 7200.0
    compression: Optional[str] = None

    def __post_init__(self):
        if self.trim_samples * 2 >= self.window_samples:
            raise ValueError(
                f"trim of {self.trim_s} s removes more than a window from every segment"
            )
        if self.margin_s <= 0 or self.block_s <= 2 * self.margin_s:
            raise ValueError(
                f"block_s ({self.block_s}) must exceed twice margin_s ({self.margin_s})"
            )
        if GWOSC_SAMPLE_RATE % self.sample_rate:
            raise ValueError(
                f"sample_rate {self.sample_rate} does not divide the GWOSC rate "
                f"{GWOSC_SAMPLE_RATE}; only integer decimation is supported"
            )

    @property
    def decimation(self) -> int:
        """Integer decimation factor from the GWOSC rate to the analysis rate."""
        return int(GWOSC_SAMPLE_RATE // self.sample_rate)

    @property
    def trim_samples(self) -> int:
        """
        Samples removed from each end of a segment, at the analysis rate.

        A whole number of samples rather than a duration, so every stored segment starts
        at the same offset from an integer GPS second. Natural segment boundaries are
        integer GPS, so all detectors then share one sample grid exactly and a window is
        aligned across the network without interpolation.
        """
        return int(round(self.trim_s * self.sample_rate))

    @property
    def window_samples(self) -> int:
        """Analysis window length in samples at the analysis rate."""
        return int(round(self.window_s * self.sample_rate))

    @property
    def minimum_segment_samples(self) -> int:
        """
        Shortest stored segment worth keeping.

        A segment that cannot host one window contributes nothing to the search and
        would otherwise appear in the release as analysable time.
        """
        if self.minimum_segment_s is not None:
            return int(round(self.minimum_segment_s * self.sample_rate))
        return self.window_samples

    @property
    def arm(self) -> str:
        """Detector set as a name, e.g. ``HLV``."""
        return "".join(d[0] for d in self.detectors)

    def release_dir(self) -> Path:
        """Where the release is written."""
        if self.out_dir is not None:
            return Path(self.out_dir)
        from sage.utils.servers import get_server

        root = Path(get_server().data_root)
        name = f"{self.observing_run.lower()}_search_data_{self.dq_flag}_{self.arm}"
        return root / name

    def scratch(self) -> Path:
        """Where GWOSC source files are staged. Never the system temporary directory."""
        if self.scratch_dir is not None:
            return Path(self.scratch_dir)
        from sage.utils.servers import get_server

        return Path(get_server().work_root) / "sage_scratch" / "dataprep"

    def whole_segment_fits(self, duration_s: float) -> bool:
        """Whether a segment of this length can be conditioned in a single pass."""
        raw_gb = duration_s * GWOSC_SAMPLE_RATE * 8 / 2**30
        return raw_gb * CONDITIONING_MEMORY_FACTOR <= self.memory_budget_gb

    def validate(self) -> None:
        """Check the conditions are self-consistent before downloading anything."""
        if not self.detectors:
            raise ValueError("no detectors requested")
        if not self.observing_run:
            raise ValueError("no observing run given")
        for detector in self.detectors:
            if not detector or len(detector) < 2:
                raise ValueError(f"malformed detector name {detector!r}")
        if self.observing_run.upper().startswith("O4A") and any(
            d.startswith("V") for d in self.detectors
        ):
            raise ValueError("Virgo strain is not published for O4a")

    def as_dict(self) -> dict:
        """JSON-serialisable record of the conditions, stored with the release."""
        out = {}
        for key, value in self.__dict__.items():
            out[key] = str(value) if isinstance(value, Path) else value
        out["detectors"] = list(self.detectors)
        out["decimation"] = self.decimation
        out["trim_samples"] = self.trim_samples
        return out


# ----------------------------------------------------------------------------
# Segment selection
# ----------------------------------------------------------------------------


def run_span(observing_run: str) -> Interval:
    """GPS span of an observing run, from the GWOSC dataset registry."""
    from gwosc.datasets import run_segment

    start, end = run_segment(observing_run)
    return float(start), float(end)


def _query_timeline(run: str, flag: str, retries: int = 8) -> List[Interval]:
    """Fetch one flag's segment list, retrying through GWOSC proxy failures."""
    import urllib.request

    start, end = run_span(run)
    url = _TIMELINE_URL.format(
        run=run, flag=flag, start=int(start), duration=int(end - start)
    )
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                payload = json.load(response)
            return [(float(a), float(b)) for a, b in payload["segments"]]
        except Exception as exc:  # noqa: BLE001 - retried below, re-raised at the end
            last = exc
            time.sleep(min(60.0, 5.0 * (attempt + 1)))
    raise RuntimeError(f"could not fetch {url}: {last}")


def segment_plan(spec: SearchDataSpec) -> Dict[str, List[Interval]]:
    """
    Observing segments to store, per detector, in GPS order.

    The detector's natural on/off structure, as published, with only those segments
    dropped that are too short to host a single analysis window once trimmed. Known
    events are kept: a search is judged on recovering them.

    ``spec.gps_range`` clips the plan to a span of the run, which builds a subset of the
    release for development against real strain without fetching all of it.
    """
    spec.validate()
    minimum_s = (
        spec.minimum_segment_samples + 2 * spec.trim_samples
    ) / spec.sample_rate
    plan: Dict[str, List[Interval]] = {}
    for detector in spec.detectors:
        flag = f"{detector}_{spec.dq_flag}"
        segments = _query_timeline(spec.observing_run, flag)
        if spec.gps_range is not None:
            lo, hi = spec.gps_range
            segments = [
                (max(a, lo), min(b, hi))
                for a, b in segments
                if min(b, hi) > max(a, lo)
            ]
        kept = [(a, b) for a, b in segments if (b - a) >= minimum_s]
        plan[detector] = sorted(kept)
    return plan


def livetime_budget(spec: SearchDataSpec) -> Dict[str, object]:
    """
    Analysable time under these conditions, and what each step costs.

    Reports the observing time, then what is lost to segments too short to analyse and
    to the trim at every segment edge, so the livetime a search will report can be
    justified before any data is fetched. Also reports the source files to fetch and the
    bytes to store, which is what determines whether the build is affordable.
    """
    spec.validate()
    minimum_s = (
        spec.minimum_segment_samples + 2 * spec.trim_samples
    ) / spec.sample_rate
    out: Dict[str, object] = {"observing_run": spec.observing_run, "detectors": {}}
    total_files = 0
    total_bytes = 0
    for detector in spec.detectors:
        segments = _query_timeline(spec.observing_run, f"{detector}_{spec.dq_flag}")
        raw_s = sum(b - a for a, b in segments)
        kept = [(a, b) for a, b in segments if (b - a) >= minimum_s]
        kept_s = sum(b - a for a, b in kept)
        stored_samples = sum(
            int(round((b - a) * spec.sample_rate)) - 2 * spec.trim_samples
            for a, b in kept
        )
        files = set()
        for a, b in kept:
            files.update(_files_spanning(a, b))
        nbytes = stored_samples * 4
        total_files += len(files)
        total_bytes += nbytes
        out["detectors"][detector] = {
            "segments_published": len(segments),
            "segments_kept": len(kept),
            "observing_s": raw_s,
            "kept_s": kept_s,
            "short_segment_loss_s": raw_s - kept_s,
            "trim_loss_s": 2 * len(kept) * spec.trim_samples / spec.sample_rate,
            "stored_samples": stored_samples,
            "stored_s": stored_samples / spec.sample_rate,
            "stored_bytes": nbytes,
            "source_files": len(files),
            "longest_segment_s": max((b - a for a, b in kept), default=0.0),
            "whole_pass_fraction": (
                sum(1 for a, b in kept if spec.whole_segment_fits(b - a)) / len(kept)
                if kept
                else 1.0
            ),
        }
    out["source_files"] = total_files
    out["stored_bytes"] = total_bytes
    return out


def _file_start(gps: float) -> int:
    """GPS start of the GWOSC file containing ``gps``."""
    return (int(math.floor(gps)) // GWOSC_FILE_DURATION_S) * GWOSC_FILE_DURATION_S


def _files_spanning(gps_start: float, gps_end: float) -> List[int]:
    """
    Starts of every GWOSC file needed to cover ``[gps_start, gps_end)``.

    A natural segment spans a median of four files and up to forty-one, so this is a
    list rather than the segment's own file and at most one neighbour.

    The end is exclusive, and the test for that is made in integer seconds: a GPS time is
    of order 1.2e9, where one unit in the last place of a float64 is 2.4e-7 s, so nudging
    the end by a small epsilon does nothing and a segment ending exactly on a boundary
    would fetch a whole extra file to read no samples from it.
    """
    first = int(math.floor(gps_start)) // GWOSC_FILE_DURATION_S
    last = (int(math.ceil(gps_end)) - 1) // GWOSC_FILE_DURATION_S
    return [
        index * GWOSC_FILE_DURATION_S for index in range(first, max(first, last) + 1)
    ]


def file_url(detector: str, observing_run: str, file_start: int) -> str:
    """
    GWOSC 4 kHz strain file URL for a detector and file start.

    Constructed rather than looked up, since resolving nearly ten thousand files through
    the API would cost as many round trips. The directory component is not always the one
    the API reports -- the archive groups files into directories on a boundary that is not
    a function of the file start alone -- but the server serves the file either way.
    :func:`resolve_file_url` is the fallback for the case where it does not.
    """
    parent = (int(file_start) // 65536) * 65536
    tag = f"{observing_run}_4KHZ_R1"
    return (
        f"https://gwosc.org/archive/data/{tag}/{parent}/"
        f"{detector[0]}-{detector}_GWOSC_{tag}-{int(file_start)}-"
        f"{GWOSC_FILE_DURATION_S}.hdf5"
    )


def resolve_file_url(detector: str, observing_run: str, file_start: int) -> str:
    """
    Ask GWOSC where a file actually is.

    Used only when the constructed URL is not found, so the build does not depend on the
    archive's directory convention staying as it is.
    """
    from gwosc.locate import get_urls

    name = (
        f"{detector[0]}-{detector}_GWOSC_{observing_run}_4KHZ_R1-"
        f"{int(file_start)}-{GWOSC_FILE_DURATION_S}.hdf5"
    )
    urls = get_urls(
        detector,
        int(file_start),
        int(file_start) + GWOSC_FILE_DURATION_S,
        sample_rate=4096,
    )
    for url in urls:
        if url.endswith(name):
            return url
    raise RuntimeError(
        f"GWOSC does not publish {name} for {detector} in {observing_run}"
    )


# ----------------------------------------------------------------------------
# Source file staging
# ----------------------------------------------------------------------------


class SlowTransfer(Exception):
    """Raised to abandon a transfer that is running far below the achievable rate."""


#: Substrings identifying a failure that is the service's, not ours, and that waiting
#: out is the correct response to. The site reaches GWOSC through an HTTP proxy which
#: returns 503 when it is saturated, and a 503 there is indistinguishable at this level
#: from GWOSC itself being briefly down; both clear on their own.
_TRANSIENT_MARKERS = (
    "proxy",
    "tunnel connection failed",
    "service unavailable",
    "max retries exceeded",
    "connection reset",
    "connection refused",
    "connection aborted",
    "timed out",
    "timeout",
    "incompleteread",
    "chunked",
    "temporarily unavailable",
    "bad gateway",
    "gateway time-out",
    "remote end closed",
)


def is_transient(exc: BaseException) -> bool:
    """
    Whether a failure is worth waiting out rather than giving up on.

    Recognised by type where the type is specific enough, and by message otherwise:
    ``requests`` wraps proxy and pool failures in generic ``ConnectionError`` whose only
    distinguishing detail is the string.
    """
    import requests

    if isinstance(exc, SlowTransfer):
        return True
    if isinstance(
        exc,
        (
            requests.exceptions.ProxyError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        response = getattr(exc, "response", None)
        return response is not None and response.status_code in (429, 500, 502, 503, 504)
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


class SourceFiles:
    """
    Staged GWOSC strain files on local disk.

    Files are fetched once, read by slice rather than in full, and evicted least
    recently used so the staging area stays bounded whatever the segment length. A
    background pool fetches ahead of the cursor, which is what keeps the network busy
    while a segment is being conditioned.

    Transfer rate from GWOSC varies by more than an order of magnitude between otherwise
    identical requests, and a connection that starts slow stays slow, so a transfer
    running below ``min_rate_mb_s`` after a grace period is abandoned and retried rather
    than ridden out. The floor is relaxed on each attempt, so a file that is genuinely
    only available slowly is still fetched instead of being retried indefinitely.
    """

    def __init__(
        self,
        scratch: Path,
        observing_run: str,
        workers: int = 4,
        max_files: int = 24,
        min_rate_mb_s: float = 2.0,
        stall_grace_s: float = 20.0,
        outage_budget_s: float = 7200.0,
    ):
        self.root = Path(scratch) / observing_run
        self.root.mkdir(parents=True, exist_ok=True)
        self.observing_run = observing_run
        self.max_files = max(2, int(max_files))
        self.min_rate_bytes_s = float(min_rate_mb_s) * 1e6
        self.stall_grace_s = float(stall_grace_s)
        self.outage_budget_s = float(outage_budget_s)
        self.abandoned = 0
        self.outage_waited_s = 0.0
        self._workers = max(1, int(workers))
        self._session = self._make_session()
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self._inflight: Dict[Tuple[str, int], object] = {}
        self._resident: "OrderedDict[Tuple[str, int], Path]" = OrderedDict()
        self._lock = threading.Lock()
        self.bytes_fetched = 0

    def _make_session(self):
        """
        Session whose adapter retries transport failures beneath ``requests``.

        Mirrors :meth:`sage.data.primer.get_data_release.DataReleaseDownloader
        ._make_retry_session`: the connection pool must be at least as large as the
        worker count or urllib3 discards connections, and 5xx are retried at the adapter
        so a brief proxy failure never reaches the caller.
        """
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retry = Retry(
            total=6,
            connect=6,
            read=6,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "HEAD"]),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        pool = max(32, self._workers * 2)
        adapter = HTTPAdapter(
            max_retries=retry, pool_connections=pool, pool_maxsize=pool
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _reset_session(self) -> None:
        """
        Discard the session and build a new one.

        A proxy outage leaves pooled connections that are open but dead; reusing them
        fails instantly and looks like a fresh outage, so the pool is rebuilt before
        waiting rather than after.
        """
        try:
            self._session.close()
        except Exception:  # noqa: BLE001 - a failed close must not mask the outage
            pass
        self._session = self._make_session()

    def close(self) -> None:
        """Shut the fetch pool down."""
        self._pool.shutdown(wait=False, cancel_futures=True)
        try:
            self._session.close()
        except Exception:  # noqa: BLE001
            pass

    def _path(self, detector: str, file_start: int) -> Path:
        return self.root / f"{detector}-{int(file_start)}.hdf5"

    def _download(self, detector: str, file_start: int, retries: int = 6) -> Path:
        """
        Fetch one file to the staging area, atomically.

        Transport failures are waited out rather than given up on. The site reaches
        GWOSC through a proxy that returns 503 under load, and an outage of a few
        minutes is routine; a build measured in days should sit through one rather than
        abandon a release that is most of the way finished. Waiting is capped by
        ``outage_budget_s`` in total, so a service that is genuinely gone still ends the
        run instead of hanging until the wall clock.
        """
        target = self._path(detector, file_start)
        if target.exists():
            return target
        url = file_url(detector, self.observing_run, file_start)
        partial = target.with_suffix(".part")
        last = None
        resolved = False
        attempt = 0
        waited = 0.0
        while True:
            floor = self.min_rate_bytes_s / (attempt + 1)
            attempt += 1
            try:
                # Separate connect and read timeouts: a dead socket must not hold a
                # worker for the whole transfer budget.
                with self._session.get(url, stream=True, timeout=(30, 120)) as response:
                    if response.status_code == 404 and not resolved:
                        # The constructed directory was wrong for this file; ask GWOSC
                        # once, then carry on with the answer.
                        resolved = True
                        url = resolve_file_url(detector, self.observing_run, file_start)
                        continue
                    response.raise_for_status()
                    started = time.monotonic()
                    seen = 0
                    with open(partial, "wb") as handle:
                        for chunk in response.iter_content(chunk_size=1 << 22):
                            handle.write(chunk)
                            seen += len(chunk)
                            elapsed = time.monotonic() - started
                            if elapsed > self.stall_grace_s and seen / elapsed < floor:
                                raise SlowTransfer(
                                    f"{seen / elapsed / 1e6:.2f} MB/s after "
                                    f"{elapsed:.0f} s"
                                )
                os.replace(partial, target)
                with self._lock:
                    self.bytes_fetched += target.stat().st_size
                return target
            except Exception as exc:  # noqa: BLE001 - classified and retried below
                last = exc
                partial.unlink(missing_ok=True)

                if isinstance(exc, SlowTransfer):
                    # Not a failure of the service; a fresh connection is usually fast,
                    # so reconnect at once. Bounded by `retries`, after which whatever
                    # rate is on offer is accepted (the floor relaxes each attempt).
                    with self._lock:
                        self.abandoned += 1
                    if attempt <= retries:
                        continue
                    raise RuntimeError(f"could not fetch {url}: {last}") from exc

                if not is_transient(exc):
                    # A missing file or a malformed request will not fix itself.
                    raise RuntimeError(f"could not fetch {url}: {last}") from exc

                if waited >= self.outage_budget_s:
                    raise RuntimeError(
                        f"could not fetch {url} after waiting {waited / 60:.0f} min "
                        f"for the service to return: {last}"
                    ) from exc

                # Escalating wait, capped, so a short blip costs seconds and a long
                # outage is not polled to death.
                wait = min(600.0, 15.0 * 2 ** min(attempt, 6))
                wait = min(wait, self.outage_budget_s - waited)
                self._reset_session()
                with self._lock:
                    self.outage_waited_s += wait
                waited += wait
                print(
                    f"    GWOSC unavailable ({type(exc).__name__}); waiting "
                    f"{wait:.0f} s then retrying {detector}-{file_start} "
                    f"({waited / 60:.0f}/{self.outage_budget_s / 60:.0f} min spent)",
                    flush=True,
                )
                time.sleep(wait)

    def prefetch(self, detector: str, file_starts: Sequence[int]) -> None:
        """
        Queue background fetches for files that will be needed soon.

        Never queues more than the staging area holds. Fetching further ahead than that
        would evict files before they were read and fetch them a second time.
        """
        with self._lock:
            room = self.max_files - len(self._inflight)
            for file_start in list(file_starts)[: max(0, room)]:
                key = (detector, int(file_start))
                if key in self._inflight or self._path(*key).exists():
                    continue
                self._inflight[key] = self._pool.submit(self._download, *key)

    def acquire(self, detector: str, file_start: int) -> Path:
        """Path to a staged file, fetching or awaiting it as needed."""
        key = (detector, int(file_start))
        with self._lock:
            future = self._inflight.pop(key, None)
        if future is not None:
            future.result()
        path = self._path(*key)
        if not path.exists():
            path = self._download(*key)
        with self._lock:
            self._resident[key] = path
            self._resident.move_to_end(key)
        self._evict()
        return path

    def _evict(self) -> None:
        """Drop least recently used files once the staging area is over budget."""
        with self._lock:
            while len(self._resident) > self.max_files:
                _, path = self._resident.popitem(last=False)
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def read(self, detector: str, gps_start: float, gps_end: float) -> np.ndarray:
        """
        Raw strain over ``[gps_start, gps_end)`` at the GWOSC rate, as float64.

        Assembled across however many files the span touches, reading only the samples
        asked for rather than whole files. Refuses a span containing unset samples:
        GWOSC pads the times a detector was not observing with NaN, so a NaN here would
        mean the segment list and the strain disagree, and one NaN propagates through
        the whole conditioning pass.
        """
        import h5py

        pieces = []
        for file_start in _files_spanning(gps_start, gps_end):
            path = self.acquire(detector, file_start)
            lo = max(gps_start, float(file_start))
            hi = min(gps_end, float(file_start + GWOSC_FILE_DURATION_S))
            with h5py.File(path, "r") as handle:
                dataset = handle["strain/Strain"]
                duration = float(handle["meta/Duration"][()])
                rate = len(dataset) / duration
                if rate != GWOSC_SAMPLE_RATE:
                    raise ValueError(
                        f"{path} is sampled at {rate} Hz, expected {GWOSC_SAMPLE_RATE}"
                    )
                i0 = int(round((lo - file_start) * rate))
                i1 = int(round((hi - file_start) * rate))
                pieces.append(dataset[i0:i1].astype(np.float64))
        raw = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        expected = int(round((gps_end - gps_start) * GWOSC_SAMPLE_RATE))
        if raw.size != expected:
            raise ValueError(
                f"{detector} [{gps_start}, {gps_end}) assembled {raw.size} samples, "
                f"expected {expected}"
            )
        bad = int(np.isnan(raw).sum())
        if bad:
            raise ValueError(
                f"{detector} [{gps_start}, {gps_end}) contains {bad} unset samples; "
                f"the {self.observing_run} segment list and the strain disagree"
            )
        return raw


# ----------------------------------------------------------------------------
# Conditioning
# ----------------------------------------------------------------------------


def condition(
    raw: np.ndarray,
    old_sample_rate: float,
    new_sample_rate: float,
    low_frequency_cutoff: float,
) -> np.ndarray:
    """
    Resample and high pass one contiguous stretch, returning float32.

    The same two operations the training releases apply, in the same order and through
    the same implementations, so the two datasets are conditioned identically and differ
    only in how the timeline was cut.
    """
    from pycbc.filter import highpass as pycbc_highpass
    from pycbc.filter.resample import resample_to_delta_t
    from pycbc.types import TimeSeries

    series = TimeSeries(raw, delta_t=1.0 / old_sample_rate)
    resampled = resample_to_delta_t(series, delta_t=1.0 / new_sample_rate)
    return pycbc_highpass(resampled, low_frequency_cutoff).numpy().astype(np.float32)


def _iter_conditioned(
    files: SourceFiles,
    detector: str,
    gps_start: float,
    gps_end: float,
    spec: SearchDataSpec,
    whole: bool,
) -> Iterator[np.ndarray]:
    """
    Conditioned float32 for one segment, in order, before trimming.

    A single pass when ``whole``, otherwise overlapping blocks whose margins are
    conditioned and then discarded. Block and margin lengths are whole numbers of
    decimated samples, so the two paths land on the same output grid.
    """
    from pycbc import DYN_RANGE_FAC

    factor = spec.decimation
    n_raw = int(round((gps_end - gps_start) * GWOSC_SAMPLE_RATE))
    if whole:
        block_raw, margin_raw = n_raw, 0
    else:
        block_raw = int(spec.block_s * GWOSC_SAMPLE_RATE) // factor * factor
        margin_raw = int(spec.margin_s * GWOSC_SAMPLE_RATE) // factor * factor

    for lo in range(0, n_raw, block_raw):
        hi = min(lo + block_raw, n_raw)
        plo = max(0, lo - margin_raw)
        phi = min(n_raw, hi + margin_raw)
        raw = files.read(
            detector,
            gps_start + plo / GWOSC_SAMPLE_RATE,
            gps_start + phi / GWOSC_SAMPLE_RATE,
        )
        conditioned = condition(
            raw, GWOSC_SAMPLE_RATE, spec.sample_rate, spec.low_frequency_cutoff
        )
        del raw
        head = (lo - plo) // factor
        interior = conditioned[head : head + (hi - lo) // factor]
        yield np.ascontiguousarray(interior * DYN_RANGE_FAC, dtype=np.float32)


# ----------------------------------------------------------------------------
# Release writing
# ----------------------------------------------------------------------------

_SIDECAR_FIELDS = (
    "segment_index",
    "detector",
    "observing_run",
    "gps_start",
    "gps_end",
    "sample_rate",
    "nsamples",
    "dtype",
    "endianness",
    "sample_start_idx",
    "dataset",
    "checksum",
    "checksum_algorithm",
    "dyn_range_fac",
    "noise_low_freq_cutoff",
    "conditioning",
    "source_files",
)


def sidecar_path(release_dir: Path, detector: str, observing_run: str) -> Path:
    """Segment index written beside the strain, in the training-release schema."""
    return Path(release_dir) / f"data_{detector}_{observing_run}_segments.json"


def strain_path(release_dir: Path, detector: str, observing_run: str) -> Path:
    """Monolithic strain file for one detector."""
    return Path(release_dir) / f"data_{detector}_{observing_run}.h5"


def _write_sidecar(path: Path, records: List[dict]) -> None:
    """Replace the sidecar atomically so a reader never sees a partial index."""
    partial = path.with_suffix(".json.part")
    partial.write_text(json.dumps(records, indent=1), encoding="utf-8")
    os.replace(partial, path)


def _existing_index(handle) -> List[dict]:
    """Segments already complete in an open release file."""
    if "index" not in handle:
        return []
    return [json.loads(row) for row in handle["index"].asstr()[:]]


def _prune_incomplete(handle) -> None:
    """
    Remove strain datasets that no completed index entry refers to.

    A segment's dataset is written before its index entry, so anything past the end of
    the index is the tail of an interrupted run and must not be treated as data.
    """
    done = {record["dataset"] for record in _existing_index(handle)}
    group = handle.require_group("segments")
    for name in list(group):
        if f"segments/{name}" not in done:
            del group[name]


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------


def prepare(
    spec: SearchDataSpec,
    detectors: Optional[Sequence[str]] = None,
    resume: bool = True,
    progress: bool = True,
) -> Dict[str, Path]:
    """
    Build a search-grade release.

    Written to its own directory; no existing dataset is touched. Segments are stored in
    GPS order in one file per detector, so a segment's index is its position in time and
    the sidecar is sorted -- unlike the training releases, where segments were written in
    parallel-completion order.

    Safe to interrupt. Each segment's strain is written before its index entry, so a
    resumed build discards at most the segment it was working on and continues from the
    index.
    """
    spec.validate()
    release = spec.release_dir()
    release.mkdir(parents=True, exist_ok=True)
    (release / "spec.json").write_text(
        json.dumps(spec.as_dict(), indent=1), encoding="utf-8"
    )

    plan = segment_plan(spec)
    wanted = tuple(detectors) if detectors else spec.detectors
    files = SourceFiles(
        spec.scratch(),
        spec.observing_run,
        workers=spec.download_workers,
        max_files=spec.cache_files,
        min_rate_mb_s=spec.min_rate_mb_s,
        stall_grace_s=spec.stall_grace_s,
        outage_budget_s=spec.outage_budget_s,
    )
    written: Dict[str, Path] = {}
    try:
        for detector in wanted:
            written[detector] = _prepare_detector(
                spec, detector, plan[detector], files, resume, progress
            )
    finally:
        files.close()

    _write_master(release, spec, wanted)
    return written


def _prepare_detector(
    spec: SearchDataSpec,
    detector: str,
    segments: List[Interval],
    files: SourceFiles,
    resume: bool,
    progress: bool,
) -> Path:
    """Build one detector's file. Returns its path."""
    import h5py
    from pycbc import DYN_RANGE_FAC

    release = spec.release_dir()
    target = strain_path(release, detector, spec.observing_run)
    sidecar = sidecar_path(release, detector, spec.observing_run)

    mode = "a" if (resume and target.exists()) else "w"
    with h5py.File(target, mode) as handle:
        if mode == "a":
            _prune_incomplete(handle)
        records = _existing_index(handle)
        handle.require_group("segments")
        handle.attrs.update(
            {
                "detector": detector,
                "observing_run": spec.observing_run,
                "dq_flag": spec.dq_flag,
                "sample_rate": spec.sample_rate,
                "dyn_range_fac": float(DYN_RANGE_FAC),
                "noise_low_freq_cutoff": spec.low_frequency_cutoff,
                "trim_samples": spec.trim_samples,
                "chunked": False,
                "overlapping": False,
                "segments_gps_ordered": True,
            }
        )

        done = len(records)
        # A resumed build continues by position, so the plan it continues into must be
        # the plan it started from. Re-querying a flag that has since been revised would
        # otherwise silently shift every remaining segment.
        for record, (gps_start, gps_end) in zip(records, segments):
            stored = gps_start + spec.trim_samples / spec.sample_rate
            if abs(record["gps_start"] - stored) > 0.5 / spec.sample_rate:
                raise ValueError(
                    f"{detector} segment {record['segment_index']} was written at "
                    f"{record['gps_start']} but the current plan plans it at {stored}; "
                    "the segment list has changed since the build started"
                )
        cursor = (
            records[-1]["sample_start_idx"] + records[-1]["nsamples"] if records else 0
        )
        todo = segments[done:]
        if progress:
            total_s = sum(b - a for a, b in todo)
            print(
                f"{detector}: {len(todo)} segments to write "
                f"({total_s / 86400:.3f} d), {done} already complete"
            )

        for offset, (gps_start, gps_end) in enumerate(todo):
            index = done + offset
            files.prefetch(
                detector,
                [
                    f
                    for a, b in segments[index : index + 3]
                    for f in _files_spanning(a, b)
                ],
            )
            record = _write_segment_resiliently(
                handle, spec, detector, index, gps_start, gps_end, cursor, files
            )
            records.append(record)
            cursor += record["nsamples"]
            _rewrite_index(handle, records)
            _write_sidecar(sidecar, [_public(r) for r in records])
            if progress:
                print(
                    f"  [{index + 1}/{len(segments)}] {gps_start:.0f}-{gps_end:.0f} "
                    f"({(gps_end - gps_start) / 3600:.2f} h, {record['conditioning']}, "
                    f"{record['nsamples'] * 4 / 2**30:.2f} GiB)",
                    flush=True,
                )
    return target


def _write_segment_resiliently(
    handle,
    spec: SearchDataSpec,
    detector: str,
    index: int,
    gps_start: float,
    gps_end: float,
    sample_start_idx: int,
    files: SourceFiles,
) -> dict:
    """
    Write one segment, sitting out any service outage rather than ending the run.

    The fetcher already waits out an outage per file; this is the layer above, for the
    case where the outage outlasts even that. A segment is self-contained -- its dataset
    is rebuilt from scratch on each attempt and its index entry is written only on
    success -- so retrying one costs the segment and nothing else.

    Segments are written in GPS order and their sample index is contiguous, so a failed
    segment cannot be skipped and filled in later: that would reorder the release. The
    choice is therefore to wait or to stop, and stopping is safe because the build
    resumes from the index.
    """
    waited = 0.0
    attempt = 0
    while True:
        try:
            return _write_segment(
                handle, spec, detector, index, gps_start, gps_end, sample_start_idx, files
            )
        except Exception as exc:  # noqa: BLE001 - classified immediately below
            attempt += 1
            if not is_transient(exc) or waited >= spec.outage_budget_s:
                raise
            wait = min(900.0, 60.0 * 2 ** min(attempt, 5))
            wait = min(wait, spec.outage_budget_s - waited)
            waited += wait
            print(
                f"    segment {index} interrupted ({type(exc).__name__}: "
                f"{str(exc)[:120]}); retrying in {wait:.0f} s "
                f"({waited / 60:.0f}/{spec.outage_budget_s / 60:.0f} min spent)",
                flush=True,
            )
            time.sleep(wait)


def _write_segment(
    handle,
    spec: SearchDataSpec,
    detector: str,
    index: int,
    gps_start: float,
    gps_end: float,
    sample_start_idx: int,
    files: SourceFiles,
) -> dict:
    """Condition and store one segment, returning its index record."""
    from pycbc import DYN_RANGE_FAC

    trim = spec.trim_samples
    n_total = int(round((gps_end - gps_start) * spec.sample_rate))
    n_out = n_total - 2 * trim
    name = f"segments/{index:06d}"
    whole = spec.whole_segment_fits(gps_end - gps_start)

    if name in handle:
        del handle[name]
    dataset = handle.create_dataset(
        name,
        shape=(n_out,),
        dtype=np.float32,
        chunks=(min(n_out, 1 << 18),),
        compression=spec.compression,
    )

    digest = hashlib.sha256()
    produced = 0  # conditioned samples seen, before trimming
    stored = 0
    for piece in _iter_conditioned(files, detector, gps_start, gps_end, spec, whole):
        lo, hi = produced, produced + piece.size
        produced = hi
        # Drop the leading and trailing trim, which may fall inside any block.
        keep_lo, keep_hi = max(lo, trim), min(hi, trim + n_out)
        if keep_hi > keep_lo:
            block = piece[keep_lo - lo : keep_hi - lo]
            dataset[stored : stored + block.size] = block
            digest.update(np.ascontiguousarray(block).tobytes())
            stored += block.size
    if stored != n_out:
        raise ValueError(
            f"{detector} segment {index} produced {stored} samples, expected {n_out}"
        )

    record = {
        "segment_index": index,
        "detector": detector,
        "observing_run": spec.observing_run,
        "gps_start": gps_start + trim / spec.sample_rate,
        "gps_end": gps_end - trim / spec.sample_rate,
        "sample_rate": spec.sample_rate,
        "nsamples": n_out,
        "dtype": "float32",
        "endianness": "<",
        "sample_start_idx": sample_start_idx,
        "dataset": name,
        "checksum": digest.hexdigest(),
        "checksum_algorithm": "sha256",
        "dyn_range_fac": float(DYN_RANGE_FAC),
        "noise_low_freq_cutoff": spec.low_frequency_cutoff,
        "conditioning": "whole" if whole else "blocked",
        "source_files": len(_files_spanning(gps_start, gps_end)),
    }
    dataset.attrs.update({k: v for k, v in record.items() if k != "dataset"})
    return record


def _public(record: dict) -> dict:
    """Sidecar view of an index record, in field order."""
    return {key: record[key] for key in _SIDECAR_FIELDS if key in record}


def _rewrite_index(handle, records: List[dict]) -> None:
    """Replace the in-file index with the completed segments."""
    import h5py

    if "index" in handle:
        del handle["index"]
    rows = np.array([json.dumps(_public(r)) for r in records], dtype=object)
    handle.create_dataset("index", data=rows, dtype=h5py.string_dtype())
    handle.attrs["n_segments"] = len(records)
    handle.attrs["total_samples"] = sum(r["nsamples"] for r in records)
    # Flush at segment granularity so a job killed at a time limit leaves a readable
    # file. Anything written past the index is an interrupted segment and is pruned on
    # resume; without the flush the whole file could be lost instead.
    handle.flush()


def _write_master(release: Path, spec: SearchDataSpec, detectors: Sequence[str]) -> None:
    """
    One file presenting the whole release, linking each detector's strain.

    HDF5 external links, so ``master["H1/segments/000000"]`` reads without the caller
    knowing the layout, while each detector stays a separate file that can be built,
    resumed and copied on its own.
    """
    import h5py

    master = release / f"data_{spec.observing_run}.h5"
    with h5py.File(master, "w") as handle:
        handle.attrs.update(
            {
                "observing_run": spec.observing_run,
                "dq_flag": spec.dq_flag,
                "detectors": list(detectors),
                "sample_rate": spec.sample_rate,
            }
        )
        for detector in detectors:
            name = strain_path(release, detector, spec.observing_run).name
            handle[detector] = h5py.ExternalLink(name, "/")


# ----------------------------------------------------------------------------
# Reading and verification
# ----------------------------------------------------------------------------


def load_release_segments(release_dir: str | Path, detector: str, observing_run: str):
    """Segment records of a built release, as :class:`sage.search.segments.Segment`."""
    from sage.search.segments import load_segments

    return load_segments(sidecar_path(Path(release_dir), detector, observing_run))


def read_segment(
    release_dir: str | Path,
    detector: str,
    observing_run: str,
    segment_index: int,
    start: int = 0,
    stop: Optional[int] = None,
) -> np.ndarray:
    """Read samples from one stored segment."""
    import h5py

    path = strain_path(Path(release_dir), detector, observing_run)
    with h5py.File(path, "r") as handle:
        dataset = handle[f"segments/{segment_index:06d}"]
        return dataset[start : (dataset.shape[0] if stop is None else stop)]


def verify(
    release_dir: str | Path, spec: SearchDataSpec, checksums: bool = False
) -> Dict[str, object]:
    """
    Check a built release against the conditions it claims.

    Confirms every planned segment is present at the right length, that stored segments
    do not overlap, and optionally that each segment's samples still match the checksum
    recorded when it was written.
    """
    import h5py

    release = Path(release_dir)
    plan = segment_plan(spec)
    out: Dict[str, object] = {"release_dir": str(release), "detectors": {}, "ok": True}
    for detector in spec.detectors:
        path = strain_path(release, detector, spec.observing_run)
        report: Dict[str, object] = {"path": str(path), "problems": []}
        if not path.exists():
            report["problems"].append("strain file missing")
            out["detectors"][detector] = report
            out["ok"] = False
            continue
        with h5py.File(path, "r") as handle:
            records = _existing_index(handle)
            expected = plan[detector]
            report["segments"] = len(records)
            report["segments_planned"] = len(expected)
            if len(records) != len(expected):
                report["problems"].append(
                    f"{len(records)} of {len(expected)} segments written"
                )
            for record, (gps_start, gps_end) in zip(records, expected):
                want = (
                    int(round((gps_end - gps_start) * spec.sample_rate))
                    - 2 * spec.trim_samples
                )
                if record["nsamples"] != want:
                    report["problems"].append(
                        f"segment {record['segment_index']} holds "
                        f"{record['nsamples']} samples, expected {want}"
                    )
            ordered = all(
                records[i]["gps_end"] <= records[i + 1]["gps_start"]
                for i in range(len(records) - 1)
            )
            if not ordered:
                report["problems"].append("segments overlap or are out of GPS order")
            report["livetime_s"] = sum(r["nsamples"] for r in records) / spec.sample_rate
            report["blocked_segments"] = sum(
                1 for r in records if r.get("conditioning") == "blocked"
            )
            if checksums:
                bad = []
                for record in records:
                    digest = hashlib.sha256()
                    dataset = handle[record["dataset"]]
                    for lo in range(0, dataset.shape[0], 1 << 22):
                        digest.update(
                            np.ascontiguousarray(dataset[lo : lo + (1 << 22)]).tobytes()
                        )
                    if digest.hexdigest() != record["checksum"]:
                        bad.append(record["segment_index"])
                report["checksum_failures"] = bad
                if bad:
                    report["problems"].append(f"{len(bad)} checksum failures")
        if report["problems"]:
            out["ok"] = False
        out["detectors"][detector] = report
    return out


def known_event_coverage(
    release_dir: str | Path,
    observing_run: str,
    detectors: Sequence[str] = ("H1", "L1"),
) -> Dict[str, object]:
    """
    Which published events for this run are present in a release.

    A search cannot recover an event whose time is absent, and recovery of known events
    is the primary evidence that a pipeline works, so this is the first thing to check
    about any release intended for a search.
    """
    from sage.search.segments import coincident_intervals

    release = Path(release_dir)
    segments = {
        detector: load_release_segments(release, detector, observing_run)
        for detector in detectors
    }
    coincident = coincident_intervals(segments)
    events = _published_events(observing_run)
    covered, missing = [], []
    for name, gps in sorted(events.items(), key=lambda kv: kv[1]):
        inside = any(lo <= gps <= hi for lo, hi in coincident)
        (covered if inside else missing).append({"name": name, "gps": gps})
    return {
        "observing_run": observing_run,
        "detectors": list(detectors),
        "n_events": len(events),
        "covered": covered,
        "missing": missing,
        "coincident_s": sum(hi - lo for lo, hi in coincident),
    }


def _published_events(observing_run: str) -> Dict[str, float]:
    """Confident events inside a run's span, from the GWOSC event API."""
    from gwosc.datasets import event_gps, find_datasets

    start, end = run_span(observing_run)
    out = {}
    for name in find_datasets(type="event"):
        try:
            gps = float(event_gps(name))
        except Exception:  # noqa: BLE001 - not every dataset entry carries a GPS
            continue
        if start <= gps <= end:
            out[name] = gps
    return out


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------


def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a search-grade strain release from GWOSC."
    )
    parser.add_argument("--run", default="O3a", help="observing run, e.g. O3a")
    parser.add_argument("--detectors", nargs="+", default=["H1", "L1", "V1"])
    parser.add_argument("--flag", default="DATA", help="data-quality flag to select on")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--scratch-dir", default=None)
    parser.add_argument(
        "--gps-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("START", "END"),
        help="build only this span of the run, for verification against real strain",
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=8.0,
        help="peak memory one conditioning pass may use; larger keeps more "
        "segments on the single-pass path",
    )
    parser.add_argument("--block-s", type=float, default=2048.0)
    parser.add_argument("--margin-s", type=float, default=32.0)
    parser.add_argument("--workers", type=int, default=4, help="parallel file fetches")
    parser.add_argument("--cache-files", type=int, default=24)
    parser.add_argument(
        "--min-rate-mb-s",
        type=float,
        default=2.0,
        help="abandon and retry a transfer running below this rate",
    )
    parser.add_argument("--stall-grace-s", type=float, default=20.0)
    parser.add_argument(
        "--outage-budget-s",
        type=float,
        default=7200.0,
        help="how long to keep waiting out a service outage before giving up",
    )
    parser.add_argument(
        "--budget", action="store_true", help="report the cost and exit"
    )
    parser.add_argument("--verify", action="store_true", help="check a built release")
    parser.add_argument("--checksums", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def redirect_temporary_files(scratch: Path) -> None:
    """
    Point every library's temporary and cache directory into the scratch area.

    The default is a small root filesystem shared by the whole node, and the libraries
    on this path stage whole strain files. Redirecting is done before any of them is
    imported or first used, since ``tempfile`` resolves its directory once and caches it.
    """
    import tempfile

    scratch = Path(scratch)
    for variable, name in (
        ("TMPDIR", "tmp"),
        ("TEMP", "tmp"),
        ("TMP", "tmp"),
        ("MPLCONFIGDIR", "matplotlib"),
        ("XDG_CACHE_HOME", "cache"),
        ("ASTROPY_CACHE_DIR", "astropy"),
    ):
        target = scratch / name
        target.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(target)
    tempfile.tempdir = str(scratch / "tmp")


def main(argv: Optional[list] = None) -> int:
    """Report the cost, build the release, or verify one."""
    args = _parse_args(argv)
    spec = SearchDataSpec(
        observing_run=args.run,
        detectors=tuple(args.detectors),
        dq_flag=args.flag,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        scratch_dir=Path(args.scratch_dir) if args.scratch_dir else None,
        gps_range=tuple(args.gps_range) if args.gps_range else None,
        memory_budget_gb=args.memory_budget_gb,
        block_s=args.block_s,
        margin_s=args.margin_s,
        download_workers=args.workers,
        cache_files=args.cache_files,
        min_rate_mb_s=args.min_rate_mb_s,
        stall_grace_s=args.stall_grace_s,
        outage_budget_s=args.outage_budget_s,
    )
    redirect_temporary_files(spec.scratch())
    if args.budget:
        print(json.dumps(livetime_budget(spec), indent=1))
        return 0
    if args.verify:
        report = verify(spec.release_dir(), spec, checksums=args.checksums)
        print(json.dumps(report, indent=1))
        return 0 if report["ok"] else 1
    written = prepare(spec, resume=not args.no_resume)
    for detector, path in written.items():
        print(f"{detector}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
