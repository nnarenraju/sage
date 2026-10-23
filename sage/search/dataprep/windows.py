"""Analysis windows over the time every detector can host one.

Reads each detector's segment list and gives every window's reference
GPS start and, per detector, its segment index and start sample.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

Interval = Tuple[float, float]


def load_segments(release_dir: Path, run: str, detector: str) -> List[dict]:
    """Segment records of one detector in a strain release.

    Parameters
    ----------
    release_dir : Path
        Release directory.
    run : str
        Observing run.
    detector : str
        Detector.

    Returns
    -------
    list of dict
        Sidecar records in GPS order.
    """
    sidecar = release_dir / f"data_{detector}_{run}_segments.json"
    records = json.loads(sidecar.read_text())
    return sorted(records, key=lambda record: record["gps_start"])


def intersect(first: List[Interval], second: List[Interval]) -> List[Interval]:
    """Intersection of two sorted lists of disjoint intervals.

    Parameters
    ----------
    first, second : list of tuple of float
        Sorted, non-overlapping intervals.

    Returns
    -------
    list of tuple of float
        Non-empty overlaps, in order.
    """
    overlaps = []
    i = j = 0
    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])
        if end > start:
            overlaps.append((start, end))

        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return overlaps


def hostable_starts(records: List[dict], window_samples: int) -> List[Interval]:
    """Start times at which a whole window fits inside one segment.

    Parameters
    ----------
    records : list of dict
        Sidecar records of one detector, in GPS order.
    window_samples : int
        Window length in samples.

    Returns
    -------
    list of tuple of float
        Earliest and latest window start per segment.
    """
    window_s = window_samples / records[0]["sample_rate"]
    return [
        (record["gps_start"], record["gps_end"] - window_s)
        for record in records
        if record["nsamples"] >= window_samples
    ]


def lattice(
    records: List[dict],
    allowed: List[Interval],
    stride_samples: int,
) -> np.ndarray:
    """GPS start of every window allowed on the reference segments.

    As in sgwc-1: windows start at the beginning of each allowed interval
    and step by the stride while the whole window still fits.

    Parameters
    ----------
    records : list of dict
        Sidecar records of the reference detector, in GPS order.
    allowed : list of tuple of float
        Intervals in which a window may start.
    stride_samples : int
        Step between window starts in samples.

    Returns
    -------
    numpy.ndarray
        Window start GPS times, in order.

    Raises
    ------
    ValueError
        If an interval edge is not on a sample of the reference segment.
    """
    rate = records[0]["sample_rate"]
    segment_starts = np.array([record["gps_start"] for record in records])

    pieces = [np.zeros(0)]
    for start, end in allowed:
        host = int(np.searchsorted(segment_starts, start, side="right")) - 1
        first = (start - segment_starts[host]) * rate
        last = (end - segment_starts[host]) * rate
        if first % 1 or last % 1:
            raise ValueError(f"interval [{start}, {end}] is not on the sample grid")

        samples = np.arange(int(first), int(last) + 1, stride_samples)
        pieces.append(segment_starts[host] + samples / rate)
    return np.concatenate(pieces)


def locate(records: List[dict], gps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Segment index and start sample of each window in one detector.

    Parameters
    ----------
    records : list of dict
        Sidecar records of the detector, in GPS order.
    gps : numpy.ndarray
        Window start GPS times.

    Returns
    -------
    segment : numpy.ndarray
        Segment index of each window.
    sample : numpy.ndarray
        Start sample of each window within its segment.
    """
    rate = records[0]["sample_rate"]
    segment_starts = np.array([record["gps_start"] for record in records])
    segment_indices = np.array([record["segment_index"] for record in records])

    position = np.searchsorted(segment_starts, gps + 0.5 / rate, side="right") - 1
    sample = np.rint((gps - segment_starts[position]) * rate).astype(np.int64)
    return segment_indices[position], sample


def analysis_windows(
    segments: Dict[str, List[dict]],
    window_samples: int,
    stride_samples: int = 204,
    reference: Optional[str] = None,
    lags: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Start of every analysis window in every detector.

    With ``lags``, a detector's data is read at the window start plus
    its lag. The analysed time is the number of windows times the stride.

    Parameters
    ----------
    segments : dict of str to list of dict
        Sidecar records per detector, in GPS order.
    window_samples : int
        Window length in samples.
    stride_samples : int
        Step between window starts in samples.
    reference : str, optional
        Detector whose segments set the lattice; the first by default.
    lags : dict of str to float, optional
        Time lag in seconds per non-reference detector; zero lag when
        omitted.

    Returns
    -------
    dict of str to numpy.ndarray
        ``gps`` holds the window start times. ``<detector>_segment``,
        ``<detector>_sample`` and ``<detector>_gps`` hold each detector's
        segment index, start sample and start time.

    Raises
    ------
    ValueError
        If the stride is not between one sample and one window.
    """
    if not 0 < stride_samples <= window_samples:
        raise ValueError(f"stride of {stride_samples} samples is invalid")
    reference = reference or next(iter(segments))
    lags = lags or {}

    allowed = None
    for detector, records in segments.items():
        lag = lags.get(detector, 0.0)
        hostable = [
            (start - lag, end - lag)
            for start, end in hostable_starts(records, window_samples)
        ]
        allowed = hostable if allowed is None else intersect(allowed, hostable)

    gps = lattice(segments[reference], allowed, stride_samples)

    windows = {"gps": gps}
    for detector, records in segments.items():
        detector_gps = gps + lags.get(detector, 0.0)
        segment, sample = locate(records, detector_gps)
        windows[f"{detector}_segment"] = segment
        windows[f"{detector}_sample"] = sample
        windows[f"{detector}_gps"] = detector_gps
    return windows
