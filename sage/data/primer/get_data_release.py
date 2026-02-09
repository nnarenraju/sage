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

Documentation: NULL

"""

# General
import os
import time
import h5py
import json
import hashlib
import warnings

# Utilities
import numpy as np
import urllib.request

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

# Multiprocessing
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppressing LAL warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Signal processing (gwpy)
from gwpy.timeseries import TimeSeries

# Signal processing (pycbc)
from pycbc import DYN_RANGE_FAC

# LOCAL
from sage.dsp.filters import pycbc_downsample
from sage.core.logger import get_logger, setup_logging

setup_logging("logs")
logger = get_logger(__name__)


@dataclass(frozen=True)
class DownloadConfig:
    # Immutable MP-safe pickleable download config
    sample_rate: float
    trim: float
    noise_low_freq_cutoff: float
    max_retries: int
    delay: float


def validate_segment(
    bin_path,
    seg_meta,
    *,
    strict=True,
):
    algo = seg_meta["checksum_algo"]

    with open(bin_path, "rb") as f:
        f.seek(seg_meta["byte_offset"])
        raw = f.read(seg_meta["byte_length"])

    h = hashlib.new(algo)
    h.update(raw)
    digest = h.hexdigest()

    ok = digest == seg_meta["checksum"]

    if not ok and strict:
        raise IOError(f"Checksum mismatch for segment {seg_meta['segment_index']}")

    return ok


def validate_all_segments(bin_path, metadata):
    failures = []

    with open(bin_path, "rb") as f:
        for seg in metadata:
            f.seek(seg["byte_offset"])
            raw = f.read(seg["byte_length"])

            h = hashlib.new(seg["checksum_algo"])
            h.update(raw)

            if h.hexdigest() != seg["checksum"]:
                failures.append(seg["segment_index"])

    return failures


def file_checksum(path, algo="sha256", block=1 << 20):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


class DataReleaseDownloader:

    def __init__(
        self,
        segments_metadata,
        save_parent_dir: str,
        noise_low_freq_cutoff: float = 15.0,
        minimum_segment_duration: float = 22.0,
        corrupt_trim_length: float = 0.2,
        max_download_retries: int = 10,
        retry_delay: float = 0.5,
        num_workers: int = 4,
        make_monolithic_file: bool = True,
        save_bin: bool = False,
        sample_rate: float = 2048.0,
    ):

        # Timeseries params
        self.sample_rate = sample_rate
        self.noise_low_freq_cutoff = noise_low_freq_cutoff
        self.trim = corrupt_trim_length
        self.minimum_segment_duration = minimum_segment_duration

        # Download params
        self.max_retries = max_download_retries
        self.delay = retry_delay
        self.num_workers = num_workers

        # Save params
        self.save_parent_dir = save_parent_dir
        self.monolithic = make_monolithic_file
        if not self.monolithic:
            logger.warning("N segment files not accepted for training Sage")
            logger.warning("Please use make_monolithic_file=True if training")
            logger.warning("Reason: Computational overhead")

        # Segments structured array
        self.full_metadata = segments_metadata

        # Save download config for safe MP
        # This is immutable and pickleable
        self.dcfg = self._return_download_config()

        # Save data in binary format instead
        self.save_bin = save_bin
        self._bin_metadata = []
        self._bin_sample_cursor = 0
        self._bin_byte_cursor = 0

    def __enter__(self):
        pass

    def __exit__(self):
        pass

    @staticmethod
    def _fetch_data(cfg, det, b0, b1):
        """Fetch a small slice of data to check availability."""
        last_exception = None

        # TODO: Generalise sage.core.errors --> "safe_call" to accommodate this
        for ntry in range(cfg.max_retries):
            try:
                data = TimeSeries.fetch_open_data(det, b0, b1, cache=1)
                return data, True
            except Exception as e:
                logger.warning(
                    f"Chunk {ntry} failed. Retrying ({ntry}/{cfg.max_retries})..."
                )
                last_exception = e
                time.sleep(cfg.delay)

        # If we get here, all retries failed
        if last_exception != None:
            logger.info(f"Tried {ntry}/{cfg.max_retries} times. Aborting.")
            # bubble up the original error
            logger.error(f"Failed to fetch {det} {b0}-{b1}: {last_exception}")
            return None, False

    @staticmethod
    def _get_detector_data(cfg, n, b0, b1, det):
        """Download detector data from GWOSC

        Args:
            args (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Download data from GWOSC
        data, fetch_okay = DataReleaseDownloader._fetch_data(cfg, det, b0, b1)
        # Handle error case
        if not fetch_okay:
            return n, None, {}

        # Process only if fetch succeeded
        old_sample_rate = 1.0 / data.dt.value
        data = pycbc_downsample(
            data.value,
            old_sample_rate,
            cfg.sample_rate,
            cfg.trim,
            cfg.noise_low_freq_cutoff,
        )
        # Apply a dynamic range factor for storage
        # NOTE: Remember to reverse this before passing to Sage
        data = data * DYN_RANGE_FAC

        metadata = {
            "gps_start": b0 + cfg.trim,
            "gps_end": b1 - cfg.trim,
            "trim": int(round(cfg.trim * cfg.sample_rate)),
            "nsamples": len(data),
            "old_sample_rate": old_sample_rate,
            "sample_rate": cfg.sample_rate,
        }
        return n, data, metadata

    def _checksum_array(self, data: np.ndarray, algo="sha256"):
        """
        Compute checksum over raw bytes of a NumPy array.
        """
        h = hashlib.new(algo)
        h.update(memoryview(data))
        return h.hexdigest()

    def _bin_open(self, filename):
        filepath = self.save_dir / filename
        return open(filepath, "wb")

    def _save_segment_bin(self, fh, data):
        """
        Append raw samples to an open binary file.
        Returns (nsamples, nbytes, written_dtype).
        """
        if data is None or not isinstance(data, np.ndarray):
            return 0, 0, None

        # Force canonical little-endian dtype
        dt = np.dtype(data.dtype).newbyteorder("<")
        data = np.ascontiguousarray(data.astype(dt, copy=False))

        raw = data.tobytes(order="C")
        fh.write(raw)
        checksum = hashlib.sha256(raw).hexdigest()

        return data.size, len(raw), dt, checksum

    def _return_download_config(self):
        """Return download config dict for MP runs"""
        return DownloadConfig(
            sample_rate=self.sample_rate,
            trim=self.trim,
            noise_low_freq_cutoff=self.noise_low_freq_cutoff,
            max_retries=self.max_retries,
            delay=self.delay,
        )

    def _save_metadata(self, hf, group_name, metadata):
        """
        Write a numpy structured array to an HDF5 file.

        This function explicitly converts arrays of fixed-length NumPy strings (S or U)
        to lists of native Python strings (str) before writing, ensuring compatibility
        with h5py's variable-length UTF-8 dtype for all records in the array.

        """

        # Create metadata group
        grp = hf.require_group(group_name)

        # --- Write simple numeric fields directly ---
        grp.create_dataset("start_time", data=metadata["start_time"])
        grp.create_dataset("end_time", data=metadata["end_time"])

        # --- Write string fields as standalone UTF-8 datasets ---
        # Define the target dtype for the HDF5 file: variable-length UTF-8
        dt_str = h5py.string_dtype(encoding="utf-8")

        detector_list = [str(x) for x in metadata["detector"]]
        grp.create_dataset("detector", data=detector_list, dtype=dt_str)

        # Data quality flag
        flag_list = [str(x) for x in metadata["flag"]]
        grp.create_dataset("flag", data=flag_list, dtype=dt_str)

        # Observing run
        observing_run_list = [str(x) for x in metadata["observing_run"]]
        grp.create_dataset("observing_run", data=observing_run_list, dtype=dt_str)

        # --- Store segments data ---
        seg_grp = grp.require_group("segments")

        # And store a simple index array
        N = len(metadata)
        seg_index = np.zeros(N, dtype="i8")

        for i in range(N):
            # The 'segments' field contains nested arrays (objects) for each record.
            segs = np.asarray(metadata["segments"][i])
            seg_grp.create_dataset(f"{i:05d}", data=segs)
            seg_index[i] = i

        grp.create_dataset("segments_index", data=seg_index)

    def _savepath_handling(self, dirname):
        """Make savepath safely"""
        # Make the save directory
        self.save_dir = Path(self.save_parent_dir) / dirname
        if not os.path.exists(self.save_dir):
            self.save_dir.mkdir(parents=True, exist_ok=False)

    def _h5py_mkfile(self, filename):
        # Make and persist open the h5py file
        filepath = self.save_dir / filename
        return h5py.File(filepath, "w")

    def _save_segment(self, hf, idx, data, det, run, chunk_metadata):
        """Save data into hdf5 dataset"""
        if data is None or not isinstance(data, np.ndarray):
            return

        seg_grp = hf.require_group("segments")
        # TODO: Padded names go up to max 99999 segments
        # This should be good enough; generalise this later if needed
        name = f"{idx:05d}"

        dset = seg_grp.create_dataset(
            name,
            data=data,
            dtype=data.dtype,
            compression="gzip",
            chunks=True,
        )

        # Attach metadata for *this* chunk
        for k, v in chunk_metadata.items():
            dset.attrs[k] = v

        dset.attrs["detector"] = det.encode("utf-8")
        dset.attrs["run"] = run.encode("utf-8")
        dset.attrs["noise_low_freq_cutoff"] = self.noise_low_freq_cutoff

    def _fetcher(self, segments, det, run):
        """
        Download GWOSC segments for a detector and store either:
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

        logger.info(
            f"Fetching GWOSC data for detector {det} ({run}) "
            f"using {self.num_workers} worker(s)"
        )

        # Setup monolithic file
        if self.monolithic and not self.save_bin:
            # Make save dir
            self._savepath_handling(f"data_release")
            # Make save file
            hf = self._h5py_mkfile(f"data_{det}_{run}.h5")
            # Store full metadata ONCE for the full dataset
            self._save_metadata(hf, "metadata", self.full_metadata)

        elif self.monolithic and self.save_bin:
            self._savepath_handling("data_release")
            bin_fh = self._bin_open(f"data_{det}_{run}.bin")
            # reset cursors
            self._bin_metadata = []
            self._bin_sample_cursor = 0
            self._bin_byte_cursor = 0

        else:
            # Make save dir
            self._savepath_handling(f"data_release_{det}_{run}")

        # Download (MP or non-MP)
        # Split segment downloads into separate tasks
        tasks = ((self.dcfg, i, b0, b1, det) for i, (b0, b1) in enumerate(segments))

        if self.num_workers > 1:
            # Setup mp tasks
            with mp.Pool(self.num_workers) as pool, tqdm(total=len(segments)) as pbar:
                pbar.set_description("MP-DET_SCIENCE_DATA GWOSC")
                for n, data, metadata in pool.starmap(
                    DataReleaseDownloader._get_detector_data, tasks
                ):
                    # NOTE: Always so save_chunk outside in main process
                    # DO NOT write let workers write to same HDF5
                    if self.save_bin:
                        nsamp, nbytes, dt, checksum = self._save_segment_bin(
                            bin_fh, data
                        )

                        seg_meta = {
                            "segment_index": int(n),
                            "detector": det,
                            "observing_run": run,
                            "gps_start": metadata["gps_start"],
                            "gps_end": metadata["gps_end"],
                            "sample_rate": metadata["sample_rate"],
                            "nsamples": nsamp,
                            "dtype": dt.name,
                            "endianness": dt.byteorder,
                            "sample_start_idx": self._bin_sample_cursor,
                            "byte_offset": self._bin_byte_cursor,
                            "byte_length": nbytes,
                            "checksum": checksum,
                            "checksum_algorithm": "sha256",
                            "dyn_range_fac": float(DYN_RANGE_FAC),
                            "noise_low_freq_cutoff": self.noise_low_freq_cutoff,
                        }

                        # Sanity check for truncation bugs
                        assert nbytes == nsamp * dt.itemsize

                        self._bin_metadata.append(seg_meta)
                        self._bin_sample_cursor += nsamp
                        self._bin_byte_cursor += nbytes

                    else:
                        self._save_segment(hf, n, data, det, run, metadata)

                    pbar.update()
        else:
            with tqdm(total=len(segments)) as pbar:
                pbar.set_description("DET_SCIENCE_DATA GWOSC")
                for args in tasks:
                    n, data, metadata = DataReleaseDownloader._get_detector_data(*args)
                    hf = self._h5py_mkfile(f"data_{det}_{run}_chunk_{n}.hdf")
                    self._save_segment(hf, n, data, det, run, metadata)
                    hf.close()
                    pbar.update()

        # Close monolithic output file
        if self.monolithic and not self.save_bin:
            hf.close()

        elif self.monolithic and self.save_bin:
            bin_fh.close()

            meta_path = self.save_dir / f"data_{det}_{run}_segments.json"
            with open(meta_path, "w") as f:
                json.dump(self._bin_metadata, f, indent=2)

        elif not self.monolithic:
            metadata_path = Path(self.save_parent_dir) / "full_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(self.full_metadata, f, indent=2)

    def _validate_segments(self, segments, det, run):
        """Validate segments based on trimmed duration and good download"""
        det_start = segments[:, 0]
        det_end = segments[:, 1]
        self.durations = det_end - det_start
        # Include corrupt_rmlength in durations
        self.durations = self.durations - (2.0 * self.trim)

        # Get valid mask based on minimum segment duration
        duration_mask = self.durations >= self.minimum_segment_duration

        # Prepare all edge checks
        edge_times = []
        for idx, (start, end) in enumerate(zip(det_start, det_end)):
            # Get 1 second near the edges to check segment validity
            edge_times.append((idx, start, start + 1))
            edge_times.append((idx, end - 1, end))

        # Get validity mask for segment availability
        edge_ok = np.ones(segments.shape[0], dtype=bool)
        # Parallel fetch
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._fetch_data, self.dcfg, det, t0, t1): idx
                for (idx, t0, t1) in edge_times
            }
            for foo in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Validating segments in {det}-{run}",
            ):
                idx = futures[foo]  # get segment index
                _, ok = foo.result()  # True/False from fetch_data
                if not ok:
                    edge_ok[idx] = False

        # Create the final mask after duration and edge checks
        return duration_mask & edge_ok

    def _clean_record(self, record):
        """Check valid segments and prune record"""
        run = record["observing_run"]
        det = record["detector"]
        segments = record["segments"]

        if segments.size == 0:
            logger.warning(f"Segments empty for {det} in {run}. Skipping.")
            return None

        logger.info(f"Downloading segments from {det} for {run}")

        # Validate if segments are okay to download
        final_mask = self._validate_segments(segments, det, run)

        # Store valid boundaries
        record["segments"] = segments[final_mask]
        total_valid_duration = self.durations[final_mask].sum()
        available_valid_duration = self.durations.sum()

        logger.info(
            f"{det} {run}: Available = {available_valid_duration}, "
            f"Valid = {total_valid_duration}."
        )
        logger.warning("Min duration & data availability might reduce valid duration.")
        return record

    ## --- Main function for end user ---

    def download(self):

        # Iterate and download records
        for record in self.full_metadata:

            # Cleanup the record
            record = self._clean_record(record)
            # Ignore if empty record
            if record == None:
                continue

            # Call fetcher to download valid data
            self._fetcher(
                record["segments"],
                record["detector"],
                record["observing_run"],
            )
