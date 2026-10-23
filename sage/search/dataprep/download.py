"""GWOSC access for building a strain release.

Observing segment lists, strain file URLs and file downloads.
"""

import math
import os
import time
from pathlib import Path
from typing import List, Tuple

import requests
from gwosc.datasets import run_segment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

Interval = Tuple[float, float]

_TIMELINE_URL = (
    "https://gwosc.org/timeline/segments/json/{run}/{flag}/{start}/{duration}/"
)


def make_session(workers: int) -> requests.Session:
    """HTTP session that retries connection failures and 5xx responses.

    Parameters
    ----------
    workers : int
        Threads sharing the session; sets the connection pool size.

    Returns
    -------
    requests.Session
        Session with a retrying adapter.
    """
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
    pool_size = max(32, 2 * workers)
    adapter = HTTPAdapter(
        max_retries=retry, pool_connections=pool_size, pool_maxsize=pool_size
    )
    session = requests.Session()
    session.mount("https://", adapter)
    return session


def run_span(run: str) -> Interval:
    """GPS span of an observing run.

    Parameters
    ----------
    run : str
        Observing run, e.g. ``O3a``.

    Returns
    -------
    tuple of float
        Start and end GPS time.
    """
    start, end = run_segment(run)
    return float(start), float(end)


def fetch_segments(
    session: requests.Session, run: str, detector: str, dq_flag: str
) -> List[Interval]:
    """Segments of one detector passing a data-quality flag.

    Parameters
    ----------
    session : requests.Session
        HTTP session.
    run : str
        Observing run.
    detector : str
        Detector, e.g. ``H1``.
    dq_flag : str
        Data-quality flag, e.g. ``DATA``.

    Returns
    -------
    list of tuple of float
        Segment start and end GPS times.
    """
    start, end = run_span(run)
    url = _TIMELINE_URL.format(
        run=run,
        flag=f"{detector}_{dq_flag}",
        start=int(start),
        duration=int(end - start),
    )
    response = session.get(url, timeout=180)
    response.raise_for_status()
    return [
        (float(segment_start), float(segment_end))
        for segment_start, segment_end in response.json()["segments"]
    ]


def files_spanning(
    gps_start: float, gps_end: float, file_duration: int = 4096
) -> List[int]:
    """Start times of the GWOSC files covering ``[gps_start, gps_end)``.

    Parameters
    ----------
    gps_start, gps_end : float
        Span in GPS time.
    file_duration : int
        Duration of one GWOSC file in seconds.

    Returns
    -------
    list of int
        File start GPS times, ascending.
    """
    first = int(math.floor(gps_start)) // file_duration
    last = (int(math.ceil(gps_end)) - 1) // file_duration
    return [index * file_duration for index in range(first, max(first, last) + 1)]


def file_url(
    detector: str, run: str, file_start: int, file_duration: int = 4096
) -> str:
    """URL of one GWOSC 4 kHz strain file.

    Parameters
    ----------
    detector : str
        Detector.
    run : str
        Observing run.
    file_start : int
        File start GPS time.
    file_duration : int
        Duration of one GWOSC file in seconds.

    Returns
    -------
    str
        File URL.
    """
    parent = (file_start // 65536) * 65536
    tag = f"{run}_4KHZ_R1"
    return (
        f"https://gwosc.org/archive/data/{tag}/{parent}/"
        f"{detector[0]}-{detector}_GWOSC_{tag}-{file_start}-{file_duration}.hdf5"
    )


def download_file(
    session: requests.Session, url: str, path: Path, attempts: int
) -> Path:
    """Download one file atomically, retrying failed transfers.

    The wait between attempts doubles from 30 s up to 600 s.

    Parameters
    ----------
    session : requests.Session
        HTTP session.
    url : str
        File URL.
    path : Path
        Destination; returned unchanged if it already exists.
    attempts : int
        Attempts before giving up.

    Returns
    -------
    Path
        Downloaded file.

    Raises
    ------
    RuntimeError
        If the file does not exist or every attempt fails.
    """
    if path.exists():
        return path
    partial = path.with_suffix(".part")
    for attempt in range(attempts):
        try:
            with session.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with open(partial, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 22):
                        handle.write(chunk)
            os.replace(partial, path)
            return path
        except requests.RequestException as error:
            partial.unlink(missing_ok=True)
            not_found = getattr(error.response, "status_code", None) == 404
            if not_found or attempt + 1 == attempts:
                raise RuntimeError(f"could not download {url}") from error
            time.sleep(min(600, 30 * 2**attempt))
