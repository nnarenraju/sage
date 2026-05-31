#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for retrying failed or missing GWOSC segment downloads.

Three public functions are provided:

``get_expected_segments``
    Re-query GWOSC and split into mini-segments exactly as the original
    download did, returning the full list of expected (b0, b1) GPS pairs.

``find_missing_segments``
    Compare that expected list against an already-downloaded segments.json
    and return records for every segment that is absent.

``retry_detector``
    End-to-end retry for one detector: discover missing/failed segments
    (from a failed_segments.json if present, otherwise via TimelineQuery)
    and append them to the existing .bin and segments.json files.
"""

import json
import math
import numpy as np
from pathlib import Path

from sage.data.primer.get_data_release import DataReleaseDownloader
from sage.data.primer.get_segments import TimelineQuery


def _compute_buffer(sample_rate, corrupt_trim_fraction=0.2):
    """Return integer-sample-aligned trim length (seconds)."""
    return math.ceil(corrupt_trim_fraction * sample_rate) / sample_rate


def get_expected_segments(
    detector,
    run,
    sample_rate=2048.0,
    corrupt_trim_fraction=0.2,
    mini_segment_duration=512.0,
    split_minimum_duration=16.0,
    prune_minimum_duration=22.0,
    rm_window_length=30.0,
):
    """
    Reconstruct the full list of expected mini-segments for a detector/run
    via TimelineQuery, using the same parameters as the original download.

    Parameters
    ----------
    detector : str
        Detector prefix, e.g. ``"H1"``.
    run : str
        Observing run label, e.g. ``"O3a"`` or ``"O3b"``.
    sample_rate : float
        Target sample rate in Hz (default 2048).
    corrupt_trim_fraction : float
        Fraction of a second trimmed from each edge to remove downsampling
        artefacts. Rounded up to the nearest integer sample count.
    mini_segment_duration : float
        Body length of each mini-segment in seconds, *before* the buffer
        is added to each edge (default 512.0).
    split_minimum_duration : float
        Minimum duration passed to ``split_into_mini_segments`` (default 16.0).
    prune_minimum_duration : float
        Minimum segment duration passed to ``prune_segments`` (default 22.0).
    rm_window_length : float
        Half-window in seconds around known events to excise (default 30.0).

    Returns
    -------
    list of (float, float)
        Pre-trim GPS (b0, b1) pairs for every expected mini-segment.
    """
    buffer = _compute_buffer(sample_rate, corrupt_trim_fraction)
    mini_seg_len = mini_segment_duration + (buffer * 2.0)

    tq = TimelineQuery(
        detector=[detector],
        observing_run=[run],
        auto_clean_empty_timelines=True,
    )
    tq.download_segments()
    tq.prune_segments(
        rm_short_segments=True,
        rm_min_duration=prune_minimum_duration,
        rm_allevents=True,
        rm_window_length=rm_window_length,
    )
    tq.split_into_mini_segments(mini_seg_len, minimum_segment_duration=split_minimum_duration)

    segs = []
    for record in tq.timeline:
        if record["detector"] == detector:
            for seg in record["segments"]:
                segs.append((float(seg[0]), float(seg[1])))
    return segs


def find_missing_segments(
    detector,
    run,
    existing_meta,
    sample_rate=2048.0,
    corrupt_trim_fraction=0.2,
    mini_segment_duration=512.0,
    split_minimum_duration=16.0,
    prune_minimum_duration=22.0,
    rm_window_length=30.0,
):
    """
    Find mini-segments expected by the GWOSC timeline but absent from
    *existing_meta* (the loaded ``*_segments.json`` content).

    The comparison un-trims the stored GPS times by *corrupt_trim_fraction*
    before matching against the pre-trim values returned by GWOSC, then
    rounds to one decimal place to absorb floating-point drift.

    Parameters
    ----------
    detector : str
    run : str
    existing_meta : list[dict]
        Contents of the ``data_{det}_{run}_segments.json`` sidecar.
    sample_rate, corrupt_trim_fraction, mini_segment_duration,
    split_minimum_duration, prune_minimum_duration, rm_window_length :
        Same meaning as in :func:`get_expected_segments`.

    Returns
    -------
    list[dict]
        One record per missing segment with keys ``segment_index``,
        ``detector``, ``observing_run``, ``gps_start``, ``gps_end``,
        ``reason``.  Ready to pass directly to :func:`retry_detector`.
    """
    buffer = _compute_buffer(sample_rate, corrupt_trim_fraction)
    print(f"[{detector}] No failed_segments.json — reconstructing via TimelineQuery ...")

    expected = get_expected_segments(
        detector, run,
        sample_rate=sample_rate,
        corrupt_trim_fraction=corrupt_trim_fraction,
        mini_segment_duration=mini_segment_duration,
        split_minimum_duration=split_minimum_duration,
        prune_minimum_duration=prune_minimum_duration,
        rm_window_length=rm_window_length,
    )

    # Un-trim stored GPS times to recover pre-trim (b0, b1) for matching
    downloaded = set()
    for m in existing_meta:
        b0 = round(m["gps_start"] - buffer, 1)
        b1 = round(m["gps_end"] + buffer, 1)
        downloaded.add((b0, b1))

    missing = []
    for b0, b1 in expected:
        if (round(b0, 1), round(b1, 1)) not in downloaded:
            missing.append({
                "segment_index": -1,
                "detector": detector,
                "observing_run": run,
                "gps_start": b0,
                "gps_end": b1,
                "reason": "missing_from_segments_json",
            })

    print(
        f"[{detector}] Timeline: {len(expected)} segments | "
        f"Downloaded: {len(downloaded)} | Missing: {len(missing)}"
    )
    return missing


def retry_detector(
    detector,
    run,
    data_dir,
    num_workers=8,
    sample_rate=2048.0,
    corrupt_trim_fraction=0.2,
    noise_low_freq_cutoff=15.0,
    max_download_retries=15,
    retry_delay=5.0,
    proxy_reset_every=50,
    proxy_reset_sleep=90.0,
    mini_segment_duration=512.0,
    split_minimum_duration=16.0,
    prune_minimum_duration=22.0,
    rm_window_length=30.0,
):
    """
    Discover and re-download all missing or failed segments for one detector,
    appending results to the existing ``.bin`` and ``segments.json`` files.

    Missing segments are located in one of two ways:

    1. If ``data_{det}_{run}_failed_segments.json`` exists in *data_dir*,
       those GPS ranges are used directly (fast path — no GWOSC query).
    2. Otherwise the GWOSC timeline is reconstructed via
       :func:`find_missing_segments` and diffed against ``segments.json``.

    On completion, ``segments.json`` is updated in-place and a
    ``*_retry_failed.json`` is written for any segments that still could
    not be fetched after *max_download_retries* attempts.

    Parameters
    ----------
    detector : str
        Detector prefix, e.g. ``"H1"``.
    run : str
        Observing run label, e.g. ``"O3a"`` or ``"O3b"``.
    data_dir : str or Path
        Directory containing ``data_{det}_{run}.bin`` and
        ``data_{det}_{run}_segments.json``.
    num_workers : int
        Number of parallel download threads (default 8).
    sample_rate : float
        Target sample rate in Hz (default 2048).
    corrupt_trim_fraction : float
    noise_low_freq_cutoff : float
    max_download_retries : int
    retry_delay : float
        Seconds to wait between retry attempts.
    proxy_reset_every : int
    proxy_reset_sleep : float
    mini_segment_duration : float
    split_minimum_duration : float
    prune_minimum_duration : float
    rm_window_length : float
    """
    data_dir = Path(data_dir)
    bin_path = data_dir / f"data_{detector}_{run}.bin"
    seg_json_path = data_dir / f"data_{detector}_{run}_segments.json"
    failed_path = data_dir / f"data_{detector}_{run}_failed_segments.json"

    if not bin_path.exists():
        print(f"[{detector}] .bin not found — skipping.")
        return
    if not seg_json_path.exists():
        print(f"[{detector}] segments.json not found — skipping.")
        return

    with open(seg_json_path) as f:
        existing_meta = json.load(f)

    if failed_path.exists():
        with open(failed_path) as f:
            failed_segs = json.load(f)
        print(f"[{detector}] {len(failed_segs)} entries in failed_segments.json")
    else:
        failed_segs = find_missing_segments(
            detector, run, existing_meta,
            sample_rate=sample_rate,
            corrupt_trim_fraction=corrupt_trim_fraction,
            mini_segment_duration=mini_segment_duration,
            split_minimum_duration=split_minimum_duration,
            prune_minimum_duration=prune_minimum_duration,
            rm_window_length=rm_window_length,
        )

    if not failed_segs:
        print(f"[{detector}] Nothing to retry.")
        return

    retry_gps = np.array(
        [[s["gps_start"], s["gps_end"]] for s in failed_segs],
        dtype=np.float64,
    )

    buffer = _compute_buffer(sample_rate, corrupt_trim_fraction)

    drd = DataReleaseDownloader(
        segments_metadata=[],
        save_parent_dir=str(data_dir.parent),
        noise_low_freq_cutoff=noise_low_freq_cutoff,
        minimum_segment_duration=split_minimum_duration,
        corrupt_trim_length=buffer,
        max_download_retries=max_download_retries,
        retry_delay=retry_delay,
        num_workers=num_workers,
        proxy_reset_every=proxy_reset_every,
        proxy_reset_sleep=proxy_reset_sleep,
        make_monolithic_file=True,
        save_bin=True,
        sample_rate=sample_rate,
    )
    drd.save_dir = data_dir

    drd.retry_and_append(
        bin_path=bin_path,
        seg_json_path=seg_json_path,
        segments_to_retry=retry_gps,
        detector=detector,
        run=run,
    )
