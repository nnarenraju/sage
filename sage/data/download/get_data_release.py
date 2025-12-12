#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : real_noise.py
Description     : Short description of the file

Created on 2025-11-06 15:00:16

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation:
    1. Take a segments structured array as input
    2. Download the updated full event list from GWOSC
    3. Create an updated segment list after removing around known events
    4. Download the segments from GWOSC with the flags taken from the array
    5. Put all segments together into one monolithic file for fast data retrieval
    6. We can also have an option to keep the files separate for weirdos

"""

# General
import time
import h5py
import json
import warnings

# Utilities
import numpy as np
import urllib.request

from tqdm import tqdm
from pathlib import Path

# Multiprocessing
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppressing LAL warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Signal processing (gwpy)
from gwpy.timeseries import TimeSeries

# Signal processing (pycbc)
from pycbc import DYN_RANGE_FAC
from pycbc.filter import highpass
from pycbc.types import TimeSeries as TS
from pycbc.filter.resample import resample_to_delta_t

# LOCAL
from sage.dsp.utils import trim_edges
from sage.core.logger import get_logger, setup_logging

setup_logging("logs")
logger = get_logger(__name__)


def _downsample(
    strain,
    old_sample_rate,
    new_sample_rate=2048.0,
    trim=0.2,
    noise_low_freq_cutoff=15.0,
):
    """Downsampling and filtering for computational reasons

    Args:
        strain (_type_): _description_
        sample_rate (float, optional): _description_. Defaults to 2048.0.

    Returns:
        _type_: _description_
    """

    # Resample and apply a highpass filter
    pycbc_strain = TS(strain, delta_t=1.0 / old_sample_rate)
    res = resample_to_delta_t(pycbc_strain, delta_t=1.0 / new_sample_rate)
    ret = highpass(res, noise_low_freq_cutoff).numpy()

    # Remove corrupted regions for edge effects
    # Defaults to 0.2, but user can be more conservative
    ret = trim_edges(ret, new_sample_rate, trim)
    return ret


def _get_detector_data(args):
    """Download detector data from GWOSC

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    (
        n,
        left_boundary,
        right_boundary,
        detector,
        new_sample_rate,
        trim,
        noise_low_freq_cutoff,
        max_retries,
        delay,
    ) = args

    last_exception = None

    for ntry in range(max_retries):
        try:
            data = TimeSeries.fetch_open_data(
                detector, left_boundary, right_boundary, cache=1
            )
        except Exception as e:
            logger.warning(f"Chunk {n} failed. Retrying ({ntry}/{max_retries})...")
            last_exception = e
            time.sleep(delay)

    if last_exception != None:
        logger.info(f"Tried {ntry}/{max_retries} times. Aborting.")
        # bubble up the original error
        logger.error(f"Chunk {n} failed: {last_exception}")
        return n, None, {}

    # Process only if fetch succeeded
    old_sample_rate = 1.0 / data.dt.value
    data = _downsample(
        data.value, old_sample_rate, new_sample_rate, trim, noise_low_freq_cutoff
    )
    # Apply a dynamic range factor for storage
    # NOTE: Remember to reverse this before passing to Sage
    data = data * DYN_RANGE_FAC

    metadata = {
        "gps_start": left_boundary + trim,
        "gps_end": right_boundary - trim,
        "trim": int(round(trim * new_sample_rate)),
        "nsamples": len(data),
        "old_sample_rate": old_sample_rate,
        "sample_rate": new_sample_rate,
    }
    return n, data, metadata


def _write_segments(hf, group_name, struct_array):
    """
    Write a numpy structured array to an HDF5 file.

    This function explicitly converts arrays of fixed-length NumPy strings (S or U)
    to lists of native Python strings (str) before writing, ensuring compatibility
    with h5py's variable-length UTF-8 dtype for all records in the array.
    """

    # For demonstration, we assume 'hf' is a valid h5py File object.

    grp = hf.require_group(group_name)

    N = len(struct_array)

    # --- Write simple numeric fields directly ---
    # These typically have float/integer dtypes and don't need casting.
    grp.create_dataset("start_time", data=struct_array["start_time"])
    grp.create_dataset("end_time", data=struct_array["end_time"])

    # --- Write string fields as standalone UTF-8 datasets ---
    # Define the target dtype for the HDF5 file: variable-length UTF-8
    dt_str = h5py.string_dtype(encoding="utf-8")

    # The list comprehension iterates over the full array (all N records)
    # and converts each record's string content to a native Python str.

    # Detector field (e.g., 'H1', 'L1')
    detector_list = [str(x) for x in struct_array["detector"]]
    grp.create_dataset("detector", data=detector_list, dtype=dt_str)

    # Flag field (e.g., 'GATED_DATA', 'GOOD_DATA')
    flag_list = [str(x) for x in struct_array["flag"]]
    grp.create_dataset("flag", data=flag_list, dtype=dt_str)

    # Observing run field (e.g., 'O2', 'O3')
    observing_run_list = [str(x) for x in struct_array["observing_run"]]
    grp.create_dataset("observing_run", data=observing_run_list, dtype=dt_str)

    # --- Store nested segments separately ---
    seg_grp = grp.require_group("segments")

    # And store a simple index array
    seg_index = np.zeros(N, dtype="i8")

    for i in range(N):
        # The 'segments' field contains nested arrays (objects) for each record.
        segs = np.asarray(struct_array["segments"][i])
        seg_grp.create_dataset(str(i), data=segs)
        seg_index[i] = i

    grp.create_dataset("segments_index", data=seg_index)


# Fetch data in MP && save data chunk
def _fetcher(
    boundaries,
    num_workers=4,
    det="",
    run="",
    parent_dir="",
    monolithic_file=True,
    trim=0.2,
    sample_rate=2048.0,
    noise_low_freq_cutoff=15.0,
    full_metadata=None,
    max_download_retries=10,
    retry_delay=0.5,
):
    """Download GWOSC segments for a detector and store either:
      - one HDF5 per chunk (default)
      - or a single monolithic HDF5 with all samples appended

    Args:
        GPS_boundaries (_type_): _description_
        num_workers (int, optional): _description_. Defaults to 4.
        det (str, optional): _description_. Defaults to "".
        run (str, optional): _description_. Defaults to "".
        parent_dir (str, optional): _description_. Defaults to "".
        monolithic_file (_type_, optional): _description_. Defaults to True.

    """

    savedir = Path(parent_dir) / f"data_{det}_{run}"
    savedir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Fetching GWOSC data for detector {det} ({run}) "
        f"using {num_workers} workers"
    )

    # Aliases
    nfs = sample_rate
    lcut = noise_low_freq_cutoff
    tasks = (
        (i, t0, t1, det, nfs, trim, lcut, max_download_retries, retry_delay)
        for i, (t0, t1) in enumerate(boundaries)
    )

    # --- Optional monolithic file setup ---
    hf_out = None
    if monolithic_file:
        monolithic_filepath = Path(parent_dir) / f"data_{det}_{run}.h5"
        hf_out = h5py.File(monolithic_filepath, "w")

        # store full metadata ONCE for the full dataset
        if full_metadata is not None:
            _write_segments(hf_out, "metadata", full_metadata)

    def save_chunk(index, data, metadata):
        """Save as individual files or into the monolithic file"""
        if data is None or not isinstance(data, np.ndarray):
            return

        # --- Monolithic mode ---
        if hf_out is not None:
            # Single-threaded write to monolithic file
            dset = hf_out.create_dataset(
                f"chunk_{index}",
                data=data,
                dtype=data.dtype,
                compression="gzip",
                chunks=True,
            )

            # attach metadata for *this* chunk
            for k, v in metadata.items():
                dset.attrs[k] = v

            dset.attrs["detector"] = det.encode("utf-8")
            dset.attrs["run"] = run.encode("utf-8")
            dset.attrs["noise_low_freq_cutoff"] = lcut

        else:
            # --- Per-file mode (default) ---
            fname = savedir / f"data_{det}_{run}_chunk_{index}.hdf"
            with h5py.File(fname, "a") as hf:
                hf.create_dataset("data", data=data, compression="gzip", chunks=True)

                # write chunk metadata
                meta_grp = hf.create_group("metadata")
                for k, v in metadata.items():
                    meta_grp.attrs[k] = v

                meta_grp.attrs["detector"] = det
                meta_grp.attrs["run"] = run
                meta_grp.attrs["noise_low_freq_cutoff"] = lcut

    # --- Parallel download ---
    if num_workers > 1:
        with mp.Pool(num_workers) as pool, tqdm(total=len(boundaries)) as pbar:
            pbar.set_description("MP-DET_SCIENCE_DATA GWOSC")
            for n, data, metadata in pool.imap_unordered(_get_detector_data, tasks):
                # NOTE: Always so save_chunk outside in main process
                # DO NOT write let workers write to same HDF5
                save_chunk(n, data, metadata)
                pbar.update()
    else:
        with tqdm(total=len(boundaries)) as pbar:
            pbar.set_description("DET_SCIENCE_DATA GWOSC")
            for args in (
                (i, t0, t1, det, nfs, trim, lcut, max_download_retries, retry_delay)
                for i, (t0, t1) in enumerate(boundaries)
            ):
                n, data, metadata = _get_detector_data(args)
                save_chunk(n, data, metadata)
                pbar.update()

    # Close monolithic output file
    if hf_out is not None:
        hf_out.close()
        logger.info(f"Wrote monolithic HDF5 to {monolithic_filepath}")

    if not monolithic_file:
        metadata_path = savedir / "full_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)


# --- Callable Function ---


def download_data(
    segments_metadata,
    save_dir: str,
    noise_low_freq_cutoff: float = 15.0,
    minimum_segment_duration: float = 22.0,
    corrupt_trim_length: float = 0.2,
    max_download_retries: int = 10,
    retry_delay: float = 0.5,
    num_workers: int = 4,
    monolithic_file: bool = True,
    sample_rate: float = 2048.0,
):

    def fetch_edge(detector, t0, t1, max_retries=10, delay=0.5):
        """Fetch a small slice of data to check availability."""
        last_exception = None

        for _ in range(max_retries):
            try:
                TimeSeries.fetch_open_data(detector, t0, t1, cache=1)
                return True
            except Exception as e:
                last_exception = e
                time.sleep(delay)

        # If we get here, all retries failed
        logger.error(f"Failed to fetch {detector} {t0}-{t1}: {last_exception}")
        return False

    # Inside your loop over runs/detectors:
    for record in segments_metadata:
        run = record["observing_run"]
        det = record["detector"]
        segments = record["segments"]
        if segments.size == 0:
            logger.warning(f"Segments empty for {det} in {run}. Skipping.")
            continue

        logger.info(f"Downloading segments from {det} for {run}")

        det_start = segments[:, 0]
        det_end = segments[:, 1]
        durations = det_end - det_start
        # Include corrupt_rmlength in durations
        durations = durations - (2.0 * corrupt_trim_length)
        # durations -= 2.0 * corrupt_trim_length

        # Get valid mask based on minimum segment duration
        duration_mask = durations >= minimum_segment_duration

        # Prepare all edge checks
        edge_times = []
        for idx, (start, end) in enumerate(zip(det_start, det_end)):
            # Get 1 second near the edges to check segment validity
            edge_times.append((idx, start, start + 1))
            edge_times.append((idx, end - 1, end))

        # Get validity mask for segment availability
        edge_ok = np.ones(segments.shape[0], dtype=bool)
        # Parallel fetch
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    fetch_edge, det, t0, t1, max_download_retries, retry_delay
                ): idx
                for (idx, t0, t1) in edge_times
            }
            for foo in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Validating segments in {det}-{run}",
            ):
                idx = futures[foo]  # get segment index
                ok = foo.result()  # True/False from fetch_edge
                if not ok:
                    edge_ok[idx] = False

        # Create the final mask after duration and edge checks
        final_mask = duration_mask & edge_ok

        # Store valid boundaries
        record["segments"] = segments[final_mask]
        total_valid_duration = durations[final_mask].sum()
        available_valid_duration = durations.sum()

        logger.info(
            f"{det} {run}: Available = {available_valid_duration}, "
            f"Valid = {total_valid_duration}."
            f"\nDuration and data availability might reduce valid duration."
        )

        ## Call fetcher to download valid data
        _fetcher(
            record["segments"],
            num_workers=num_workers,
            det=det,
            run=run,
            parent_dir=save_dir,
            monolithic_file=monolithic_file,
            trim=corrupt_trim_length,
            sample_rate=sample_rate,
            noise_low_freq_cutoff=noise_low_freq_cutoff,
            full_metadata=segments_metadata,
            max_download_retries=max_download_retries,
            retry_delay=retry_delay,
        )
