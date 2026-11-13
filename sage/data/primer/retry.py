#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Retries of failed or missing GWOSC segment downloads: the expected
mini-segments, those missing from a download, and their re-download
appended to the existing binary file.
"""

import json
import math
import numpy as np
from pathlib import Path

from sage.data.primer.get_data_release import DataReleaseDownloader
from sage.data.primer.get_segments import TimelineQuery


def _compute_buffer(sample_rate, corrupt_trim_fraction=0.2):
    """
    Edge trim in seconds, rounded up to whole samples.

    Parameters
    ----------
    sample_rate : float
        Sample rate in Hz.
    corrupt_trim_fraction : float
        Trim in seconds before the rounding.

    Returns
    -------
    float
        The trim in seconds.
    """
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
    Expected mini-segments of a detector and run, from TimelineQuery.

    With the original download's parameters, the segments match it.

    Parameters
    ----------
    detector : str
        Detector prefix, e.g. ``"H1"``.
    run : str
        Observing run label, e.g. ``"O3a"`` or ``"O3b"``.
    sample_rate : float
        Sample rate after downsampling in Hz.
    corrupt_trim_fraction : float
        Seconds trimmed from each edge, rounded up to whole samples.
    mini_segment_duration : float
        Mini-segment length in seconds before the trim is added to each
        edge.
    split_minimum_duration : float
        ``minimum_segment_duration`` of ``split_into_mini_segments``.
    prune_minimum_duration : float
        Shortest segment kept by ``prune_segments``, in seconds.
    rm_window_length : float
        Half-width in seconds of the window cut around each GWOSC event.

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
    Expected mini-segments absent from a download's segment list.

    The stored GPS times are un-trimmed, and both sides rounded to
    0.1 s, before matching.

    Parameters
    ----------
    detector : str
        Detector prefix.
    run : str
        Observing run label.
    existing_meta : list of dict
        Contents of the ``data_{det}_{run}_segments.json`` sidecar.
    sample_rate, corrupt_trim_fraction, mini_segment_duration
        As for :func:`get_expected_segments`.
    split_minimum_duration, prune_minimum_duration, rm_window_length
        As for :func:`get_expected_segments`.

    Returns
    -------
    list of dict
        One record per missing segment, with ``segment_index``,
        ``detector``, ``observing_run``, ``gps_start``, ``gps_end`` and
        ``reason``.
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

    # Stored GPS times un-trimmed to the pre-trim (b0, b1)
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
    Re-download the missing or failed segments of one detector.

    The segments come from ``data_{det}_{run}_failed_segments.json``
    if it exists, else from :func:`find_missing_segments`; they are
    appended with :meth:`DataReleaseDownloader.retry_and_append`.

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
        Number of download threads.
    sample_rate : float
        Sample rate after downsampling in Hz.
    corrupt_trim_fraction : float
        Seconds trimmed from each edge, rounded up to whole samples.
    noise_low_freq_cutoff : float
        High-pass frequency of the downsampling in Hz.
    max_download_retries : int
        Attempts at each GWOSC file download.
    retry_delay, proxy_reset_every, proxy_reset_sleep
        Passed to :class:`DataReleaseDownloader`.
    mini_segment_duration, split_minimum_duration
        As for :func:`get_expected_segments`.
    prune_minimum_duration, rm_window_length
        As for :func:`get_expected_segments`.
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
