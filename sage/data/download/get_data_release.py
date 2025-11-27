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

# Downloading
import urllib.request

# Scientific
import scipy
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt

# Utilities
from tqdm import tqdm
from sys import getsizeof

# Multiprocessing
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

# Signal processing
from gwpy.timeseries.TimeSeries import fetch_open_data
from gwpy.segments import DataQualityFlag
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

# Removing LAL warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# LOCAL
from sage.core.decorators import reference
from sage.core.logger import get_logger

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

    out = zeros(n, dtype=arr.dtype)
    out[:shift] = arr[n - shift :]
    out[shift:] = arr[: n - shift]
    return out


def _fir_zero_filter(coeff, timeseries):
    """Filter the timeseries with a set of FIR coefficients

    Parameters
    ----------
    coeff: numpy.ndarray
        FIR coefficients. Should be and odd length and symmetric.
    timeseries: pycbc.types.TimeSeries
        Time series to be filtered.

    Returns
    -------
    filtered_series: pycbc.types.TimeSeries
        Return the filtered timeseries, which has been properly shifted to account
    for the FIR filter delay and the corrupted regions zeroed out.
    """
    # apply the filter
    ## NOTE: This takes a np.ndarray and not a PyCBC timeseries
    ## So we just call this from PyCBC instead.
    series = lfilter(coeff, timeseries)

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
    data = _fir_zero_filter(filter_coefficients, timeseries)[::factor]


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

    n = int(trim * fs)

    if n == 0 or 2 * n >= len(data):
        logger.error("Trim too large for data length.")
        raise ValueError("Trim too large for data length.")

    return data[n:-n]


def _downsample(strain, sample_rate=2048.0, trim=0.2):
    """Downsampling and filtering for computational reasons

    Args:
        strain (_type_): _description_
        sample_rate (float, optional): _description_. Defaults to 2048.0.

    Returns:
        _type_: _description_
    """

    # Resample and apply an FIR filter
    res = _resample(strain, 1.0 / sample_rate)

    # Apply IIR sosfiltfilt for lowpass
    ret = _highpass_butter_sosfiltfilt(
        red, fs=sample_rate, cutoff=15.0, order=4, padlen_seconds=2.5
    ).astype(np.float32)

    # Remove corrupted regions for edge effects
    # Defaults to 0.2, but user can be more conservative
    ret = _trim_edges(ret, sample_rate, trim)
    return ret


def _get_detector_data(args):
    """Download detector data from GWOSC

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    n, left_boundary, right_boundary, detector = args

    try:
        data = fetch_open_data(detector, left_boundary, right_boundary, cache=1)
    except Exception:
        # bubble up the original error
        raise

    # Process only if fetch succeeded
    data = _downsample(data.value, 1.0 / data.dt.value, trim)
    # Apply a dynamic range factor for storage
    # NOTE: Remember to reverse this before passing to Sage
    data = data * DYN_RANGE_FAC
    return n, data


# Fetch data in MP && save data chunk
def _fetcher(
    GPS_boundaries,
    num_workers=4,
    det="",
    run="",
    parent_dir="",
    monolithic_file=None,
    trim_seconds=0.2,
    sample_rate=2048.0,
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
        monolithic_file (_type_, optional): _description_. Defaults to None.

    """

    savedir = Path(parent_dir) / f"data_{det}_{run}"
    savedir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Fetching GWOSC data for detector {det} ({run}) "
        f"using {num_workers} workers"
    )

    tasks = ((i, t0, t1, det) for i, (t0, t1) in enumerate(GPS_boundaries))

    # --- Optional monolithic file setup ---
    if monolithic_file is not None:
        monolithic_file = Path(monolithic_file)
        hf_out = h5py.File(monolithic_file, "w")
        metadata_dtype = np.dtype(
            [
                ("chunk_id", int),
                ("gps_start", float),
                ("gps_end", float),
                ("detector", "S2"),
                ("run", "S10"),
                ("nsamples", int),
            ]
        )
        metadata_list = []

    def save_chunk(n, data):
        """Save as individual files or into the monolithic file"""
        if data is None or not isinstance(data, np.ndarray):
            return

        # --- Monolithic mode ---
        if monolithic_file is not None:
            # Single-threaded write to monolithic file
            dset = hf_out.create_dataset(
                f"chunk_{n}",
                data=data,
                dtype=data.dtype,
                compression="gzip",
                chunks=True,
            )

            # attach metadata to dataset
            dset.attrs["gps_start"] = chunk["gps_start"]
            dset.attrs["gps_end"] = chunk["gps_end"]
            dset.attrs["detector"] = chunk["det"]
            dset.attrs["run"] = chunk["run"]
            dset.attrs["nsamples"] = len(chunk["data"])

        else:
            # --- Per-file mode (default) ---
            fname = savedir / f"data_{det}_{run}_chunk_{n}.hdf"
            with h5py.File(fname, "a") as hf:
                hf.create_dataset("data", data=data, compression="gzip", chunks=True)

    # --- Parallel download ---
    if num_workers > 1:
        with mp.Pool(num_workers) as pool, tqdm(total=len(GPS_boundaries)) as pbar:
            pbar.set_description("MP-DET_SCIENCE_DATA GWOSC")
            for n, data in pool.imap_unordered(_get_detector_data, tasks):
                save_chunk(n, data)
                pbar.update()
    else:
        with tqdm(total=len(GPS_boundaries)) as pbar:
            pbar.set_description("DET_SCIENCE_DATA GWOSC")
            for args in ((i, t0, t1, det) for i, (t0, t1) in enumerate(GPS_boundaries)):
                n, data = _get_detector_data(args)
                save_chunk(n, data)
                pbar.update()

    # Close monolithic output file
    if monolithic_file is not None:
        metadata_array = np.array(metadata_list, dtype=metadata_dtype)
        hf_out.create_dataset("segments_metadata", data=metadata_array)
        hf_out.close()
        logger.info(f"\nWrote monolithic HDF5 to {monolithic_file}")


# --- Callable Function ---


def download_data(
    segments_metadata,
    noise_low_freq_cutoff: float = 15.0,
    minimum_segment_duration: float = 22.0,
    corrupt_rmlength: float = 0.2,
    max_download_retries: int = 10,
    retry_delay: float = 0.5,
):

    def fetch_edge(detector, t0, t1, max_retries=10, delay=0.5):
        """Fetch a small slice of data to check availability."""
        last_exception = None

        for _ in range(max_retries):
            try:
                fetch_open_data(detector, t0, t1, cache=1)
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
        durations -= 2.0 * corrupt_rmlength

        # Get valid mask based on minimum segment duration
        duration_mask = durations >= minimum_segment_duration

        # Prepare all edge checks
        edge_times = []
        for idx, (start, end) in enumerate(zip(det_start, det_end)):
            # Get 1 second near the edges to check segment validity
            edge_times.append((idx, start, start + 1))
            edge_times.append((idx, end - 1, end))

        # Get validity mask for segment availability
        edge_ok = np.ones(num_segments, dtype=bool)
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
                    edge_ok[i] = False

        # Create the final mask after duration and edge checks
        final_mask = duration_mask & edge_ok

        # Store valid boundaries
        record["segments"] = segments[final_mask]
        total_valid_duration = durations[final_mask].sum()
        available_valid_duration = durations.sum()

        logger.info(
            f"{det} {run}: Available = {available_valid_duration}, "
            f"Valid = {total_valid_duration}."
            f"Duration and data availability might reduce valid duration."
        )
