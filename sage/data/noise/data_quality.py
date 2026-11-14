"""
Time that noise must not be drawn from, per detector and observing run.

GWOSC time failing CBC Category 2 (and so Category 1) or holding
hardware injections, and every span of exactly-zero strain (missing
data written as zeros inside GWOSC's DATA segments).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from gwosc.datasets import run_segment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sage.config.machine import bad_time_file, segments_file, zero_runs_file
from sage.core.progress import bar
from sage.core.hardcode import _GWOSC_TIMELINE_URL

Interval = Tuple[float, float]
FLAGS = ("CBC_CAT2", "NO_CBC_HW_INJ", "NO_BURST_HW_INJ", "NO_DETCHAR_HW_INJ", "NO_STOCH_HW_INJ")


def get_segments(
    session: requests.Session, run: str, flag: str, start: int, end: int
) -> List[Interval]:
    """Segments passing one GWOSC flag, in one request.

    Parameters
    ----------
    session : requests.Session
        Retrying HTTP session.
    run : str
        Observing run.
    flag : str
        Flag with its detector, e.g. ``H1_CBC_CAT2``.
    start, end : int
        GPS span.

    Returns
    -------
    list of tuple of float
        Sorted segments.

    Raises
    ------
    requests.HTTPError
        If the request fails.
    """
    url = _GWOSC_TIMELINE_URL.format(run=run, flag=flag, start=start, duration=end - start)
    response = session.get(url, timeout=300)
    response.raise_for_status()
    return [tuple(segment) for segment in response.json()["segments"]]


def subtract(first: List[Interval], second: List[Interval]) -> List[Interval]:
    """Parts of ``first`` outside ``second``.

    Parameters
    ----------
    first, second : list of tuple of float
        Sorted, non-overlapping intervals.

    Returns
    -------
    list of tuple of float
        Remaining intervals, in order.
    """
    remaining = []
    j = 0
    for start, end in first:
        t = start
        while j < len(second) and second[j][1] <= t:
            j += 1

        k = j
        while k < len(second) and second[k][0] < end:
            if second[k][0] > t:
                remaining.append((t, second[k][0]))
            t = max(t, second[k][1])
            k += 1

        if t < end:
            remaining.append((t, end))
    return remaining


def bad_segments(run: str, detector: str) -> List[Interval]:
    """Time of a detector failing CBC CAT2 or with hardware injections.

    Each GWOSC flag lists the time that passes, so the bad time is
    ``DATA`` minus it. Continuous-wave injections run throughout and are
    not counted.

    Parameters
    ----------
    run : str
        Observing run, e.g. ``O3a``.
    detector : str
        Detector, e.g. ``H1``.

    Returns
    -------
    list of tuple of float
        Sorted, non-overlapping bad intervals.
    """
    # Retry settings of sage.search.dataprep.download.make_session.
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
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))

    start, end = run_segment(run)
    data = get_segments(session, run, f"{detector}_DATA", start, end)

    bad = []
    for flag in FLAGS:
        passing = get_segments(session, run, f"{detector}_{flag}", start, end)
        bad += subtract(data, passing)
    return merge(bad)


def merge(intervals: List[Interval]) -> List[Interval]:
    """Intervals sorted, with overlapping or touching ones joined.

    Parameters
    ----------
    intervals : list of tuple of float
        Intervals in any order.

    Returns
    -------
    list of tuple of float
        Sorted, non-overlapping intervals.
    """
    merged = []
    for first, last in sorted(intervals):
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def write_bad_time(path: Path, run: str, detector: str) -> None:
    """Save ``bad_segments`` beside a strain file, with their flags.

    Parameters
    ----------
    path : Path
        Strain ``.bin`` file.
    run : str
        GWOSC observing run, e.g. ``O3a``.
    detector : str
        Detector, e.g. ``H1``.
    """
    file = bad_time_file(path)
    part = file.with_name(f"{file.name}.part")
    part.write_text(json.dumps({"flags": list(FLAGS), "intervals": bad_segments(run, detector)}))
    os.replace(part, file)


def read_bad_time(path: Path) -> List[Interval]:
    """The bad time saved beside a strain file, zero runs included.

    The intervals of ``write_bad_time`` and ``write_zero_runs`` merged.

    Parameters
    ----------
    path : Path
        Strain ``.bin`` file.

    Returns
    -------
    list of tuple of float
        Sorted, non-overlapping bad intervals.

    Raises
    ------
    ValueError
        If the bad time was saved from other flags.
    FileNotFoundError
        If the zero runs were never saved.
    """
    file = bad_time_file(path)
    saved = json.loads(file.read_text())
    if saved["flags"] != list(FLAGS):
        raise ValueError(f"{file} holds the flags {saved['flags']}, not {list(FLAGS)}: "
                         f"move it aside, then sage download")
    zeros = zero_runs_file(path)
    if not zeros.exists():
        raise FileNotFoundError(f"{zeros} is missing: sage download writes it")
    return merge([tuple(interval) for interval in saved["intervals"] + json.loads(zeros.read_text())["intervals"]])


def zero_spans(
    samples: np.ndarray, gps_start: float, sample_rate: float
) -> List[Interval]:
    """GPS spans of exactly-zero samples.

    Parameters
    ----------
    samples : numpy.ndarray
        Strain samples.
    gps_start : float
        GPS time of the first sample.
    sample_rate : float
        Sample rate in Hz.

    Returns
    -------
    list of tuple of float
        ``[start, end)`` of each run of zeros, in order.
    """
    zero = samples == 0
    if not zero.any():
        return []

    edges = np.flatnonzero(np.diff(np.concatenate(([0], zero.view(np.int8), [0]))))
    return [
        (gps_start + first / sample_rate, gps_start + last / sample_rate)
        for first, last in zip(edges[::2].tolist(), edges[1::2].tolist())
    ]


def zero_runs(path: Path) -> List[Interval]:
    """GPS spans of exactly-zero samples in a strain file.

    Missing data is written as zeros. A piece's ``zero_spans``, found
    in its raw strain at download, are taken as recorded; a piece
    without them has its stored strain scanned.

    Parameters
    ----------
    path : Path
        Strain ``.bin`` file with its ``_segments.json`` sidecar.

    Returns
    -------
    list of tuple of float
        Sorted, non-overlapping ``[start, end)`` spans; a segment
        overlapping any of them holds a zero.
    """
    found = []
    for record in bar(json.loads(segments_file(path).read_text()), unit="piece"):
        if "zero_spans" in record:
            found += [tuple(span) for span in record["zero_spans"]]
            continue
        dtype = np.dtype(record["dtype"]).newbyteorder(record["endianness"])
        samples = np.memmap(path, dtype=dtype, mode="r", offset=record["byte_offset"], shape=(record["nsamples"],))
        found += zero_spans(samples, record["gps_start"], record["sample_rate"])

    return merge(found)


def write_zero_runs(path: Path) -> None:
    """Save ``zero_runs`` beside a strain file.

    Parameters
    ----------
    path : Path
        Strain ``.bin`` file.
    """
    file = zero_runs_file(path)
    part = file.with_name(f"{file.name}.part")
    part.write_text(json.dumps({"intervals": zero_runs(path)}))
    os.replace(part, file)


def noise_bad_segments(bins: List[str]) -> Dict[Tuple[str, str], List[Interval]]:
    """Bad time of every observing run and detector among noise files.

    Parameters
    ----------
    bins : list of str
        Noise ``.bin`` files, each with its ``_segments.json`` sidecar
        and ``_bad_time.json``.

    Returns
    -------
    dict of tuple of str to list of tuple of float
        ``read_bad_time`` keyed by ``(run, detector)``.
    """
    bad = {}
    for path in map(Path, bins):
        first = json.loads(segments_file(path).read_text())[0]
        bad[(first["observing_run"], first["detector"])] = read_bad_time(path)
    return dict(sorted(bad.items()))


def overlaps(
    starts: np.ndarray, ends: np.ndarray, bad: List[Interval]
) -> np.ndarray:
    """Spans ``[start, end)`` overlapping any bad interval, even partly.

    Parameters
    ----------
    starts, ends : numpy.ndarray
        GPS start and end of each span.
    bad : list of tuple of float
        Sorted, non-overlapping bad intervals (``read_bad_time``).

    Returns
    -------
    numpy.ndarray
        True for each overlapping span.
    """
    if not bad:
        return np.zeros(len(starts), dtype=bool)

    # The first bad interval ending after a span starts overlaps it if
    # it begins before the span ends
    bad_starts, bad_ends = np.array(bad, dtype=float).T
    first = np.searchsorted(bad_ends, starts, side="right")
    hit = first < len(bad)
    hit[hit] = bad_starts[first[hit]] < ends[hit]
    return hit


def clean(records: List[dict], bad: List[Interval]) -> np.ndarray:
    """Sidecar records that overlap no bad interval.

    Parameters
    ----------
    records : list of dict
        Segment sidecar records with ``gps_start``, ``nsamples`` and
        ``sample_rate``.
    bad : list of tuple of float
        Sorted, non-overlapping bad intervals (``read_bad_time``).

    Returns
    -------
    numpy.ndarray
        True for each clean record.
    """
    starts = np.array([record["gps_start"] for record in records], dtype=float)
    lengths = [record["nsamples"] / record["sample_rate"] for record in records]
    return ~overlaps(starts, starts + np.array(lengths), bad)
