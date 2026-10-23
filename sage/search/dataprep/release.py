"""Build a search-grade strain release from GWOSC.

One HDF5 file per detector holds one dataset per observing segment; a
JSON sidecar lists the segments. An interrupted build resumes.
"""

import contextlib
import json
import math
import os
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import requests
from pycbc import DYN_RANGE_FAC

from sage.dsp.filters import pycbc_downsample
from sage.search.dataprep.download import (
    Interval,
    download_file,
    fetch_segments,
    file_url,
    files_spanning,
    make_session,
)


def select_segments(
    segments: List[Interval],
    minimum_duration: float,
    sample_rate: float,
    gps_range: Optional[Interval] = None,
) -> List[Interval]:
    """Clip segments to a GPS range and drop those too short to store.

    Parameters
    ----------
    segments : list of tuple of float
        Segment start and end GPS times.
    minimum_duration : float
        Shortest segment kept, in seconds.
    sample_rate : float
        Stored sample rate in Hz; range edges must fall on its samples.
    gps_range : tuple of float, optional
        Keep only time inside this span.

    Returns
    -------
    list of tuple of float
        Kept segments in GPS order.

    Raises
    ------
    ValueError
        If an edge of ``gps_range`` is not on a sample.
    """
    if gps_range is not None:
        range_start, range_end = gps_range
        for edge in gps_range:
            if (edge * sample_rate) % 1:
                raise ValueError(f"gps_range edge {edge} is not on a sample")
        segments = [
            (max(start, range_start), min(end, range_end))
            for start, end in segments
            if min(end, range_end) > max(start, range_start)
        ]
    return sorted(
        (start, end) for start, end in segments if end - start >= minimum_duration
    )


def read_strain(
    staged: Dict[int, Path],
    gps_start: float,
    gps_end: float,
    source_sample_rate: float = 4096.0,
    file_duration: int = 4096,
) -> np.ndarray:
    """Raw GWOSC strain over ``[gps_start, gps_end)``.

    Parameters
    ----------
    staged : dict of int to Path
        Staged GWOSC file per file start GPS time.
    gps_start, gps_end : float
        Span in GPS time.
    source_sample_rate : float
        Sample rate of the GWOSC files in Hz.
    file_duration : int
        Duration of one GWOSC file in seconds.

    Returns
    -------
    numpy.ndarray
        Strain as float64.

    Raises
    ------
    ValueError
        If a file has the wrong sample rate, or the span has the wrong
        number of samples or contains NaN samples.
    """
    pieces = []
    for file_start in files_spanning(gps_start, gps_end, file_duration):
        span_start = max(gps_start, float(file_start))
        span_end = min(gps_end, float(file_start + file_duration))
        with h5py.File(staged[file_start], "r") as handle:
            dataset = handle["strain/Strain"]
            rate = len(dataset) / float(handle["meta/Duration"][()])
            if rate != source_sample_rate:
                raise ValueError(f"{staged[file_start]} is sampled at {rate} Hz")
            first_sample = int(round((span_start - file_start) * rate))
            last_sample = int(round((span_end - file_start) * rate))
            pieces.append(dataset[first_sample:last_sample].astype(np.float64))
    raw = np.concatenate(pieces)

    expected = int(round((gps_end - gps_start) * source_sample_rate))
    if raw.size != expected:
        raise ValueError(f"[{gps_start}, {gps_end}) has {raw.size} samples")
    n_nan = int(np.isnan(raw).sum())
    if n_nan:
        raise ValueError(f"[{gps_start}, {gps_end}) has {n_nan} NaN samples")
    return raw


def write_segment(
    handle: h5py.File,
    strain: np.ndarray,
    detector: str,
    run: str,
    index: int,
    segment: Interval,
    sample_start_idx: int,
    sample_rate: float,
    trim_samples: int,
    low_frequency_cutoff: float,
) -> dict:
    """Store conditioned strain and return its sidecar record.

    Parameters
    ----------
    handle : h5py.File
        Open strain file.
    strain : numpy.ndarray
        Conditioned strain with the trim already removed.
    detector : str
        Detector.
    run : str
        Observing run.
    index : int
        Segment position in the release.
    segment : tuple of float
        Segment start and end GPS time, before trimming.
    sample_start_idx : int
        Sample offset of the segment in the release.
    sample_rate : float
        Sample rate of ``strain`` in Hz.
    trim_samples : int
        Samples removed from each end.
    low_frequency_cutoff : float
        High-pass cutoff applied, in Hz.

    Returns
    -------
    dict
        Sidecar record.

    Raises
    ------
    ValueError
        If the strain length does not match the segment.
    """
    gps_start, gps_end = segment
    n_samples = int(round((gps_end - gps_start) * sample_rate)) - 2 * trim_samples
    if strain.size != n_samples:
        raise ValueError(f"{detector} segment {index} has {strain.size} samples")

    name = f"segments/{index:06d}"
    handle.create_dataset(name, data=strain, chunks=(min(n_samples, 1 << 18),))
    return {
        "segment_index": index,
        "detector": detector,
        "observing_run": run,
        "gps_start": gps_start + trim_samples / sample_rate,
        "gps_end": gps_end - trim_samples / sample_rate,
        "sample_rate": sample_rate,
        "nsamples": n_samples,
        "sample_start_idx": sample_start_idx,
        "dataset": name,
        "dyn_range_fac": float(DYN_RANGE_FAC),
        "noise_low_freq_cutoff": low_frequency_cutoff,
    }


def load_records(
    sidecar: Path, segments: List[Interval], sample_rate: float, trim_samples: int
) -> List[dict]:
    """Sidecar records of a partial build, checked against the segments.

    Parameters
    ----------
    sidecar : Path
        Sidecar path; may not exist yet.
    segments : list of tuple of float
        Segments to store, in GPS order.
    sample_rate : float
        Stored sample rate in Hz.
    trim_samples : int
        Samples removed from each end of a segment.

    Returns
    -------
    list of dict
        Records already written.

    Raises
    ------
    ValueError
        If a written segment does not match the segment list.
    """
    records = json.loads(sidecar.read_text()) if sidecar.exists() else []
    for record, (gps_start, _) in zip(records, segments):
        planned = gps_start + trim_samples / sample_rate
        if abs(record["gps_start"] - planned) > 0.5 / sample_rate:
            raise ValueError(f"segment {record['segment_index']} is not in the list")
    return records


def build_detector(
    session: requests.Session,
    run: str,
    detector: str,
    segments: List[Interval],
    release_dir: Path,
    scratch_dir: Path,
    sample_rate: float,
    low_frequency_cutoff: float,
    trim_samples: int,
    workers: int,
    files_ahead: int,
    download_attempts: int,
) -> Path:
    """Write, or resume writing, one detector's strain file.

    Source files are downloaded in parallel ahead of the segment being
    written and deleted once no later segment needs them.

    Parameters
    ----------
    session : requests.Session
        HTTP session.
    run : str
        Observing run.
    detector : str
        Detector.
    segments : list of tuple of float
        Segments to store, in GPS order.
    release_dir : Path
        Release directory.
    scratch_dir : Path
        Directory for downloaded source files.
    sample_rate : float
        Stored sample rate in Hz.
    low_frequency_cutoff : float
        High-pass cutoff in Hz.
    trim_samples : int
        Samples removed from each end of a segment.
    workers : int
        Parallel downloads.
    files_ahead : int
        Source files downloaded ahead of the current segment.
    download_attempts : int
        Attempts per source file.

    Returns
    -------
    Path
        Strain file.
    """
    strain_path = release_dir / f"data_{detector}_{run}.h5"
    sidecar = release_dir / f"data_{detector}_{run}_segments.json"
    records = load_records(sidecar, segments, sample_rate, trim_samples)
    n_done = len(records)
    print(f"{detector}: {len(segments) - n_done} segments to write", flush=True)

    needed = [files_spanning(start, end) for start, end in segments]
    order = sorted({file_start for files in needed[n_done:] for file_start in files})
    scratch_dir.mkdir(parents=True, exist_ok=True)
    downloads: Dict[int, Future] = {}
    queued = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        with h5py.File(strain_path, "a") as handle:
            # Delete datasets written but not indexed before a stop.
            indexed = {entry["dataset"] for entry in records}
            group = handle.require_group("segments")
            for name in list(group):
                if f"segments/{name}" not in indexed:
                    del group[name]

            for index in range(n_done, len(segments)):
                while queued < len(order) and (
                    order[queued] <= needed[index][-1] or len(downloads) < files_ahead
                ):
                    file_start = order[queued]
                    downloads[file_start] = pool.submit(
                        download_file,
                        session,
                        file_url(detector, run, file_start),
                        scratch_dir / f"{detector}-{file_start}.hdf5",
                        download_attempts,
                    )
                    queued += 1
                staged = {
                    file_start: downloads[file_start].result()
                    for file_start in needed[index]
                }

                raw = read_strain(staged, *segments[index])
                strain = pycbc_downsample(
                    raw,
                    old_sample_rate=4096.0,
                    new_sample_rate=sample_rate,
                    trim=trim_samples / sample_rate,
                    noise_low_freq_cutoff=low_frequency_cutoff,
                )
                strain = (strain * DYN_RANGE_FAC).astype(np.float32, copy=False)
                sample_start_idx = sum(entry["nsamples"] for entry in records)
                record = write_segment(
                    handle,
                    strain,
                    detector,
                    run,
                    index,
                    segments[index],
                    sample_start_idx,
                    sample_rate,
                    trim_samples,
                    low_frequency_cutoff,
                )
                records.append(record)
                handle.flush()
                partial = sidecar.with_suffix(".part")
                partial.write_text(json.dumps(records, indent=1), encoding="utf-8")
                os.replace(partial, sidecar)

                still_needed = (
                    needed[index + 1][0] if index + 1 < len(segments) else math.inf
                )
                for file_start in list(downloads):
                    if file_start < still_needed:
                        downloads.pop(file_start).result().unlink(missing_ok=True)
                print(f"  [{index + 1}/{len(segments)}] {segments[index]}", flush=True)
    finally:
        pool.shutdown(cancel_futures=True)

    for leftover in scratch_dir.glob(f"{detector}-*.hdf5"):
        leftover.unlink()
    return strain_path


def download_search_data(
    run: str,
    detectors: List[str],
    release_dir: Path,
    gps_range: Optional[Interval] = None,
    sample_rate: float = 2048.0,
    low_frequency_cutoff: float = 15.0,
    trim_s: float = 0.2,
    minimum_segment_s: float = 16.0,
    dq_flag: str = "DATA",
    workers: int = 8,
    files_ahead: int = 24,
    download_attempts: int = 10,
) -> None:
    """Download and condition search strain for every detector.

    Source files are staged in ``release_dir/scratch``, which is removed
    once empty. Calling again resumes an interrupted build.

    Parameters
    ----------
    run : str
        Observing run, e.g. ``O3a``.
    detectors : list of str
        Detectors to build.
    release_dir : Path
        Release directory.
    gps_range : tuple of float, optional
        Store only time inside this span.
    sample_rate : float
        Stored sample rate in Hz.
    low_frequency_cutoff : float
        High-pass cutoff in Hz.
    trim_s : float
        Duration removed from each end of a segment, rounded up to whole
        samples.
    minimum_segment_s : float
        Shortest stored segment after trimming.
    dq_flag : str
        Data-quality flag selecting the segments.
    workers : int
        Parallel downloads.
    files_ahead : int
        Source files downloaded ahead of the current segment.
    download_attempts : int
        Attempts per source file.
    """
    # TODO: verify the trim covers resample and high-pass edge ringing.
    trim_samples = math.ceil(trim_s * sample_rate)
    if trim_samples != trim_s * sample_rate:
        warnings.warn(
            f"trim of {trim_s} s is {trim_s * sample_rate} samples; "
            f"rounded up to {trim_samples}"
        )
    minimum_samples = round(minimum_segment_s * sample_rate) + 2 * trim_samples
    scratch_dir = release_dir / "scratch"
    release_dir.mkdir(parents=True, exist_ok=True)
    session = make_session(workers)
    for detector in detectors:
        segments = fetch_segments(session, run, detector, dq_flag)
        segments = select_segments(
            segments, minimum_samples / sample_rate, sample_rate, gps_range
        )
        build_detector(
            session,
            run,
            detector,
            segments,
            release_dir,
            scratch_dir,
            sample_rate,
            low_frequency_cutoff,
            trim_samples,
            workers,
            files_ahead,
            download_attempts,
        )
    # Kept if another build is still staging files here.
    with contextlib.suppress(OSError):
        scratch_dir.rmdir()
