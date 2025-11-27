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
    2. Download the updated full event list from GWOSC (DONE)
    3. Create an updated segment list after removing around known events (DONE)
    4. Download the segments from GWOSC with the flags taken from the array
    5. Put all segments together into one monolithic file for fast data retrieval
    6. We can also have an option to keep the files separate for weirdos

"""

# General
import os
import sys
import time
import h5py
import math
import glob
import json
import pickle
import warnings
import itertools
import functools

from pathlib import Path

# Downloading
import urllib.request

# Scientific
import scipy
import numpy as np

# Plotting
import matplotlib.pyplot as plt

# Utilities
from tqdm import tqdm
from sys import getsizeof

# Multiprocessing
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings

# Suppressing LAL warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Signal processing
from gwpy.timeseries import TimeSeries
from gwpy.segments import DataQualityFlag
from pycbc.types import TimeSeries as TS
from pycbc.filter import highpass
from pycbc.filter.resample import lfilter

from scipy.signal import (
    butter,
    sosfiltfilt,
    sosfilt,
    firwin,
    fftconvolve,
    sosfreqz,
)

# Constants
from pycbc import DYN_RANGE_FAC

# LOCAL
from sage.core.decorators import reference
from sage.core.logger import get_logger, setup_logging
from sage.core.types import SEGMENT_DTYPE

setup_logging("logs")
logger = get_logger(__name__)


# --- Obtained from PyCBC ---
# Mild modifications have been made to the original
PYCBC_PARENT_URL = "https://pycbc.org/pycbc/latest/html/_modules/"


@functools.lru_cache(maxsize=20)
def _cached_firwin(*args, **kwargs):
    """Cache the FIR filter coefficients.
    This is done if user requests for lots of segments.
    """
    return firwin(*args, **kwargs)


def _roll(arr, shift):
    """Roll an array
    Removes dependence from PyCBC timeseries

    Args:
        arr (_type_): _description_
        shift (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = len(arr)
    shift %= n
    if shift == 0:
        return arr

    out = np.zeros(n, dtype=arr.dtype)
    out[:shift] = arr[n - shift :]
    out[shift:] = arr[: n - shift]
    return out


def _fir_zero_filter(coeff, timeseries):
    """Filter the timeseries with a set of FIR coefficients

    Parameters
    ----------
    coeff: numpy.ndarray
        FIR coefficients. Should be and odd length and symmetric.
    timeseries: np.ndarray
        Time series to be filtered (not PyCBC TimeSeries object).

    Returns
    -------
    filtered_series: pycbc.types.TimeSeries
        Return the filtered timeseries, which has been properly shifted to account
    for the FIR filter delay and the corrupted regions zeroed out.
    """
    # apply the filter
    ## NOTE: This takes a np.ndarray and not a PyCBC timeseries
    ## So we just call this from PyCBC instead.
    series = lfilter(coeff, timeseries).numpy()

    # reverse the time shift caused by the filter,
    # corruption regions contain zeros
    # If the number of filter coefficients is odd, the central point *should*
    # be included in the output so we only zero out a region of len(coeff) - 1
    series[: (len(coeff) // 2) * 2] = 0
    series = _roll(series, -len(coeff) // 2)
    return series


@reference(
    os.path.join(PYCBC_PARENT_URL, "pycbc/filter/resample.html"),
    os.path.join(PYCBC_PARENT_URL, "pycbc/types/array.html#Array.roll"),
    category="documentation",
)
def _resample(strain, old_delta_t, new_delta_t):
    factor = int(round(new_delta_t / old_delta_t))
    numtaps = factor * 20 + 1

    # The kaiser window has been testing using the LDAS implementation
    # and is in the same configuration as used in the original lalinspiral
    filter_coefficients = _cached_firwin(numtaps, 1.0 / factor, window=("kaiser", 5))

    # apply the filter and decimate
    pycbc_strain = TS(strain, delta_t=new_delta_t)
    data = _fir_zero_filter(filter_coefficients, pycbc_strain)[::factor]
    return data


# --- END ---


def _ensure_1d(x):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("input must be 1D")
    return x


def _highpass_butter_sosfiltfilt(
    x,
    fs,
    cutoff=15.0,
    order=4,
    padlen_seconds=2.0,
    pad_type="reflect",
    fallback_to_sos=False,
):
    """
    Robust zero-phase highpass using Butterworth SOS + filtfilt-like forward-backward.
    NOTE: Padding reduces edge effects

    Args:
        x (array): 1D time series (float).
        fs (float): sampling rate (Hz).
        cutoff (float): cutoff frequency (Hz).
        order (int): nominal Butterworth order (sos built from order).
        padlen_seconds (float): pad length in seconds (mirror padding)
        pad_type: 'reflect' or 'odd' or 'constant' (prefer 'reflect').
        fallback_to_sos: if True and sosfiltfilt fails, do sosfilt (causal).

    Returns:
        y (np.ndarray): filtered 1D array same length as input.

    """

    x = _ensure_1d(x)
    n = x.shape[0]
    if n == 0:
        return x

    nyq = 0.5 * fs
    if cutoff <= 0 or cutoff >= nyq:
        logger.error("cutoff must be between 0 and Nyquist (fs/2)")
        raise ValueError("cutoff must be between 0 and Nyquist (fs/2)")

    # Build SOS
    sos = butter(order, cutoff / nyq, btype="highpass", output="sos")

    # compute pad length in samples (must be >= some multiple of filter transient)
    padlen = int(max(3, padlen_seconds * fs))
    # ensure padlen < n
    if padlen >= n:
        padlen = max(0, n // 2 - 1)

    # Apply padding
    if padlen > 0:
        if pad_type == "reflect":
            xp = np.pad(x, padlen, mode="reflect")
        elif pad_type == "symmetric":
            xp = np.pad(x, padlen, mode="symmetric")
        elif pad_type == "odd":
            xp = np.pad(x, padlen, mode="odd")
        elif pad_type == "constant":
            xp = np.pad(x, padlen, mode="constant", constant_values=0.0)
        else:
            logger.error("unsupported pad_type")
            raise ValueError("unsupported pad_type")
    else:
        xp = x

    # Filter with zero-phase sosfiltfilt
    try:
        y = sosfiltfilt(sos, xp)
    except Exception as exc:
        # fallback: try sosfilt (causal) if requested, otherwise re-raise
        logger.warning("sosfiltfilt failed; falling back to sosfilt if enabled")
        if fallback_to_sos:
            y = sosfilt(sos, xp)
        else:
            logger.warning("sosfilt fallback failed")
            logger.info("Set fallback_to_sos to True to enable fallback")
            raise

    # remove padding
    if padlen > 0:
        y = y[padlen : padlen + n]

    return y


def _trim_edges(data, fs, trim=0.2):
    """Trim data edges after filtering/resampling.

    Args:
        data (array): 1D time series
        fs (float): Sampling rate (Hz)
        trim (float, optional): Edge trim (seconds) for normal mode.
            - Defaults to 0.2

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    n = int(round(trim * fs))

    if n == 0 or 2 * n >= len(data):
        logger.error("Trim too large for data length.")
        raise ValueError("Trim too large for data length.")

    return data[n:-n]


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

    # Resample and apply an FIR filter
    res = _resample(strain, 1.0 / old_sample_rate, 1.0 / new_sample_rate)

    # Apply IIR sosfiltfilt for highpass
    ret = _highpass_butter_sosfiltfilt(
        res,
        fs=new_sample_rate,
        cutoff=noise_low_freq_cutoff,
        order=4,
        padlen_seconds=2.5,
    ).astype(np.float32)

    # Remove corrupted regions for edge effects
    # Defaults to 0.2, but user can be more conservative
    ret = _trim_edges(ret, new_sample_rate, trim)
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
    ) = args

    try:
        data = TimeSeries.fetch_open_data(
            detector, left_boundary, right_boundary, cache=1
        )
    except Exception as e:
        # bubble up the original error
        logger.error(f"Chunk {n} failed: {e}")
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

    nfs = sample_rate
    lcut = noise_low_freq_cutoff
    tasks = ((i, t0, t1, det, nfs, trim, lcut) for i, (t0, t1) in enumerate(boundaries))

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

            dset.attrs["detector"] = det
            dset.attrs["run"] = run
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
                (i, t0, t1, det, nfs, trim, lcut)
                for i, (t0, t1) in enumerate(boundaries)
            ):
                n, data, metadata = _get_detector_data(args)
                save_chunk(n, data, metadata)
                pbar.update()

    # Close monolithic output file
    if hf_out is not None:
        hf_out.close()
        logger.info(f"\nWrote monolithic HDF5 to {monolithic_file}")

    raise

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
        )
