"""The part of an observing run's segments a noise source draws from.

``portion`` keeps a fraction of the data in time order, cut at whole
segments (the volume method of ``split.py``); ``gps_range`` keeps all data
inside a GPS range, trimming segments at its edges.
"""

import math
from fractions import Fraction

import numpy as np

from sage.data.noise.data_quality import overlaps


def volume_cutoff(records_by_detector: dict, fraction: float) -> float:
    """
    GPS time at which the data volume of all detectors reaches a fraction.

    Segments of every detector are ordered by start; the cut is the start
    of the first segment at which the cumulative ``nsamples`` reaches
    ``fraction`` of the total, so that segment opens the later part.

    Parameters
    ----------
    records_by_detector : dict of str to list of dict
        Segment sidecar records per detector.
    fraction : float
        Fraction of the total volume, in ``(0, 1)``.

    Returns
    -------
    float
        GPS start of the cutting segment.
    """
    segments = sorted(
        (
            (record["gps_start"], record["nsamples"])
            for records in records_by_detector.values()
            for record in records
        ),
        key=lambda segment: segment[0],
    )
    total = sum(nsamples for _, nsamples in segments)
    accumulated = 0.0
    for gps_start, nsamples in segments:
        accumulated += nsamples
        if accumulated / total >= fraction:
            return gps_start
    raise ValueError(f"fraction {fraction} is not reached by the data")


def select_portion(records_by_detector: dict, portion: list) -> dict:
    """
    Keep the segments between two volume fractions, in time order.

    A segment straddling a cut is dropped, so parts of one run never share
    data. ``[0.0, 0.9]`` and ``[0.9, 1.0]`` give ``split.py``'s two parts.

    Parameters
    ----------
    records_by_detector : dict of str to list of dict
        Segment sidecar records per detector.
    portion : list of float
        ``[low, high]`` with ``0 <= low < high <= 1``.

    Returns
    -------
    dict of str to list of dict
        Kept records per detector, in their original order.
    """
    low, high = portion
    lower = -math.inf if low == 0 else volume_cutoff(records_by_detector, low)
    upper = math.inf if high == 1 else volume_cutoff(records_by_detector, high)
    return {
        detector: [r for r in records if lower <= r["gps_start"] and r["gps_end"] <= upper]
        for detector, records in records_by_detector.items()
    }


def trim(record: dict, first: int, last: int) -> dict:
    """
    A segment record restricted to samples ``[first, last)``.

    Parameters
    ----------
    record : dict
        Segment sidecar record.
    first, last : int
        Sample offsets within the segment.

    Returns
    -------
    dict
        Copy with start index, sample count and GPS span updated.
    """
    rate = Fraction(repr(record["sample_rate"]))
    start = Fraction(repr(record["gps_start"]))
    trimmed = dict(record)
    trimmed["sample_start_idx"] = record["sample_start_idx"] + first
    trimmed["nsamples"] = last - first
    trimmed["gps_start"] = float(start + first / rate)
    trimmed["gps_end"] = float(start + last / rate)
    if "byte_offset" in record:
        itemsize = np.dtype(record["dtype"]).itemsize
        trimmed["byte_offset"] = record["byte_offset"] + first * itemsize
    return trimmed


def select_gps_range(records_by_detector: dict, gps_range: list) -> dict:
    """
    Keep all data inside a GPS range, trimming segments at its edges.

    Parameters
    ----------
    records_by_detector : dict of str to list of dict
        Segment sidecar records per detector.
    gps_range : list of float
        ``[start, end]`` GPS times, ``start < end``.

    Returns
    -------
    dict of str to list of dict
        Kept, trimmed records per detector, in their original order.
    """
    range_start, range_end = (Fraction(repr(t)) for t in gps_range)
    selected = {}
    for detector, records in records_by_detector.items():
        kept = []
        for record in records:
            rate = Fraction(repr(record["sample_rate"]))
            start = Fraction(repr(record["gps_start"]))
            first = max(0, math.ceil((range_start - start) * rate))
            last = min(record["nsamples"], math.floor((range_end - start) * rate))
            if last > first:
                whole = first == 0 and last == record["nsamples"]
                kept.append(record if whole else trim(record, first, last))
        selected[detector] = kept
    return selected


def select_records(records_by_detector: dict, source: dict) -> dict:
    """
    Apply a source's ``portion`` or ``gps_range``, if it has one.

    Parameters
    ----------
    records_by_detector : dict of str to list of dict
        Segment sidecar records per detector of one run.
    source : dict
        Checked noise source; ``portion`` and ``gps_range`` are optional.

    Returns
    -------
    dict of str to list of dict
        Records the source draws from.

    Raises
    ------
    ValueError
        If both are given, either is malformed, or nothing is selected.
    """
    portion, gps_range = source.get("portion"), source.get("gps_range")
    if portion is not None and gps_range is not None:
        raise ValueError("a noise source takes portion or gps_range, not both")

    if portion is not None:
        if len(portion) != 2 or not 0 <= portion[0] < portion[1] <= 1:
            raise ValueError(f"portion must be [low, high] with 0 <= low < high <= 1, got {portion}")
        selected = select_portion(records_by_detector, portion)
    elif gps_range is not None:
        if len(gps_range) != 2 or not gps_range[0] < gps_range[1]:
            raise ValueError(f"gps_range must be [start, end] with start < end, got {gps_range}")
        selected = select_gps_range(records_by_detector, gps_range)
    else:
        return records_by_detector

    for detector, records in selected.items():
        if not records:
            raise ValueError(f"no {detector} data in portion {portion} / gps_range {gps_range}")
    return selected


def check_disjoint(first: dict, second: dict, run: str) -> None:
    """
    Check that two sources share no data of one observing run.

    Parameters
    ----------
    first, second : dict of str to list of dict
        Selected records per detector of the same run.
    run : str
        Observing run, for the message.

    Raises
    ------
    ValueError
        If any segment of ``first`` overlaps one of ``second``.
    """
    for detector, records in first.items():
        spans = [(r["gps_start"], r["gps_end"]) for r in second[detector]]
        starts = np.array([r["gps_start"] for r in records], dtype=float)
        ends = np.array([r["gps_end"] for r in records], dtype=float)
        if overlaps(starts, ends, spans).any():
            raise ValueError(f"training and validation noise share {run} {detector} data")


def summary(records: list) -> str:
    """
    One-line amount and span of a detector's selected segments.

    Parameters
    ----------
    records : list of dict
        Selected segment records.

    Returns
    -------
    str
        Segment count, days of data and first / last GPS time.
    """
    days = sum(r["nsamples"] / r["sample_rate"] for r in records) / 86400.0
    return (
        f"{len(records)} segments, {days:.2f} days, "
        f"GPS {records[0]['gps_start']}-{records[-1]['gps_end']}"
    )
