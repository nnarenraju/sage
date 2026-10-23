"""GWOSC data-quality time that noise must not be drawn from.

Time failing CBC Category 2 (and so Category 1) or holding hardware
injections, per detector and observing run.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from gwosc.datasets import run_segment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

Interval = Tuple[float, float]
_TIMELINE_URL = (
    "https://gwosc.org/timeline/segments/json/{run}/{flag}/{start}/{duration}/"
)


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
    """
    url = _TIMELINE_URL.format(run=run, flag=flag, start=start, duration=end - start)
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
    """Time of one detector failing CBC Category 2 or holding hardware injections.

    Each GWOSC flag lists the time that passes, so the bad time is ``DATA``
    minus it. Continuous-wave injections run throughout and are not counted.

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
    for flag in (
        "CBC_CAT2",
        "NO_CBC_HW_INJ",
        "NO_BURST_HW_INJ",
        "NO_DETCHAR_HW_INJ",
        "NO_STOCH_HW_INJ",
    ):
        passing = get_segments(session, run, f"{detector}_{flag}", start, end)
        bad += subtract(data, passing)

    merged = []
    for first, last in sorted(bad):
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def noise_bad_segments(bins: List[str]) -> Dict[Tuple[str, str], List[Interval]]:
    """Bad time of every observing run and detector among noise files.

    Parameters
    ----------
    bins : list of str
        Noise ``.bin`` files, each with its ``_segments.json`` sidecar.

    Returns
    -------
    dict of tuple of str to list of tuple of float
        ``bad_segments`` keyed by ``(run, detector)``.
    """
    keys = set()
    for path in map(Path, bins):
        sidecar = path.parent / f"{path.stem}_segments.json"
        first = json.loads(sidecar.read_text())[0]
        keys.add((first["observing_run"], first["detector"]))
    return {key: bad_segments(*key) for key in sorted(keys)}


def overlaps(
    starts: np.ndarray, ends: np.ndarray, bad: List[Interval]
) -> np.ndarray:
    """Spans ``[start, end)`` that overlap any bad interval, even partly.

    Parameters
    ----------
    starts, ends : numpy.ndarray
        GPS start and end of each span.
    bad : list of tuple of float
        Output of ``bad_segments``.

    Returns
    -------
    numpy.ndarray
        True for each overlapping span.
    """
    if not bad:
        return np.zeros(len(starts), dtype=bool)

    # The first bad interval ending after a span starts overlaps it if it
    # begins before the span ends.
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
        Output of ``bad_segments``.

    Returns
    -------
    numpy.ndarray
        True for each clean record.
    """
    starts = np.array([record["gps_start"] for record in records], dtype=float)
    lengths = [record["nsamples"] / record["sample_rate"] for record in records]
    return ~overlaps(starts, starts + np.array(lengths), bad)
