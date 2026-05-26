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
import re
import time
import h5py
import json
import hashlib
import warnings
import tempfile
import collections

# Utilities
import numpy as np
import urllib.request
import requests as _requests

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

# Multiprocessing
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

# GWOSC file URL resolution
from gwosc.locate import get_urls as _gwosc_get_urls

# Suppressing LAL warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Signal processing (gwpy)
from gwpy.timeseries import TimeSeries

# Signal processing (pycbc)
from pycbc import DYN_RANGE_FAC

# LOCAL
from sage.dsp.filters import pycbc_downsample

# from sage.core.logger import get_logger, setup_logging

# setup_logging("logs")
# logger = get_logger(__name__)


@dataclass(frozen=True)
class DownloadConfig:
    """
    Immutable, multiprocessing-safe download configuration.

    Frozen dataclass passed to worker processes via :mod:`multiprocessing`
    ``Pool.starmap``; must be pickleable (no open file handles, no locks).

    Attributes
    ----------
    sample_rate : float
        Target sample rate (Hz) after downsampling.
    trim : float
        Seconds of corrupt data to trim from each edge after download.
    noise_low_freq_cutoff : float
        Low-frequency cutoff (Hz) used during downsampling.
    max_retries : int
        Maximum number of download attempts before giving up on a segment.
    delay : float
        Seconds to wait between retry attempts.
    """

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
    """
    Verify the checksum of a single binary segment.

    Parameters
    ----------
    bin_path : str or Path
        Path to the ``.bin`` file.
    seg_meta : dict
        Segment metadata dict from the sidecar JSON (must contain
        ``"checksum_algo"``, ``"byte_offset"``, ``"byte_length"``,
        ``"checksum"``, and ``"segment_index"``).
    strict : bool
        If ``True`` (default), raise :exc:`IOError` on mismatch.

    Returns
    -------
    bool
        ``True`` if the checksum matches, ``False`` otherwise.
    """
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
    """
    Verify checksums for all segments in *metadata*.

    Parameters
    ----------
    bin_path : str or Path
        Path to the ``.bin`` file.
    metadata : list[dict]
        List of segment metadata dicts as produced by the sidecar JSON.

    Returns
    -------
    list[int]
        Segment indices that failed the checksum.  An empty list means all
        segments are intact.
    """
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
    """
    Compute the checksum of an entire file using streaming reads.

    Parameters
    ----------
    path : str or Path
        File to hash.
    algo : str
        Hash algorithm name (default ``"sha256"``).
    block : int
        Read block size in bytes (default 1 MiB).

    Returns
    -------
    str
        Hex-encoded digest string.
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


class DataReleaseDownloader:
    """
    Parallel GWOSC data downloader that stores validated strain segments to HDF5.

    Downloads LIGO/Virgo/KAGRA open-data strain segments from GWOSC for a
    structured-array of (detector, observing-run, segment-list) records.
    Supports both a monolithic HDF5 file and a raw binary output format,
    with optional checksum validation per segment.

    Parameters
    ----------
    segments_metadata : np.ndarray (SEGMENT_DTYPE)
        Structured array produced by :class:`~sage.data.primer.get_segments.TimelineQuery`.
    save_parent_dir : str
        Root directory under which the ``data_release/`` sub-folder is created.
    noise_low_freq_cutoff : float
        High-pass cutoff applied during downsampling (default ``15.0`` Hz).
    minimum_segment_duration : float
        Segments shorter than this (seconds, post-trim) are discarded
        (default ``22.0``).
    corrupt_trim_length : float
        Seconds trimmed from each edge to discard corrupted data (default
        ``0.2``).
    max_download_retries : int
        Maximum number of GWOSC fetch retries per segment (default ``10``).
    retry_delay : float
        Seconds between retries (default ``0.5``).
    num_workers : int
        Number of parallel worker processes (default ``4``).
    make_monolithic_file : bool
        If ``True`` (default), all segments for a (detector, run) pair are
        concatenated into a single HDF5; otherwise one file per chunk.
    save_bin : bool
        If ``True``, write raw binary (``float32 LE``) instead of HDF5, with
        a companion JSON segment index (default ``False``).
    sample_rate : float
        Target sample rate after downsampling (default ``2048.0`` Hz).
    """

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
        proxy_reset_every: int = 50,
        proxy_reset_sleep: float = 90.0,
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
        self.proxy_reset_every = proxy_reset_every
        self.proxy_reset_sleep = proxy_reset_sleep

        # Save params
        self.save_parent_dir = save_parent_dir
        self.monolithic = make_monolithic_file
        # if not self.monolithic:
        #    logger.warning("N segment files not accepted for training Sage")
        #    logger.warning("Please use make_monolithic_file=True if training")
        #    logger.warning("Reason: Computational overhead")

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
                data = TimeSeries.fetch_open_data(det, b0, b1, cache=True)
                return data, True
            except Exception as e:
                # logger.warning(
                #    f"Chunk {ntry} failed. Retrying ({ntry}/{cfg.max_retries})..."
                # )
                last_exception = e
                time.sleep(cfg.delay)

        # If we get here, all retries failed
        if last_exception != None:
            # logger.info(f"Tried {ntry}/{cfg.max_retries} times. Aborting.")
            # bubble up the original error
            # logger.error(f"Failed to fetch {det} {b0}-{b1}: {last_exception}")
            return None, False

    @staticmethod
    def _get_detector_data_unpacked(args):
        return DataReleaseDownloader._get_detector_data(*args)

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

    @staticmethod
    def _file_gps_interval(url):
        """Return (gps_start, gps_end) from a GWOSC HDF5 file URL, or (None, None)."""
        m = re.search(r"-(\d+)-(\d+)\.(hdf5?|gwf)$", url)
        if m:
            t0 = int(m.group(1))
            return t0, t0 + int(m.group(2))
        return None, None

    @staticmethod
    def _make_retry_session(connect_retries=5, pool_size=32):
        """Return a requests.Session with urllib3 retry and adequate pool size.

        pool_size must be >= num_workers to avoid "Connection pool is full"
        discards. Default of 32 comfortably covers up to 32 parallel threads.
        """
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        session = _requests.Session()
        retry = Retry(
            total=connect_retries,
            connect=connect_retries,
            read=connect_retries,
            backoff_factor=2.0,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=pool_size,
            pool_maxsize=pool_size,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    @staticmethod
    def _gwosc_file_start(t):
        """GPS start of the 4096s GWOSC file that contains time t.

        GWOSC 4096s files start at GPS times that are exact multiples of 4096.
        """
        return (int(t) // 4096) * 4096

    @staticmethod
    def _construct_file_url(det, f0, observing_run="O3b"):
        """Construct GWOSC 4KHz HDF5 URL from detector and GPS file start.

        URL pattern (verified for O3b):
          https://gwosc.org/archive/data/{run}_4KHZ_R1/{parent}/{prefix}-{det}_GWOSC_{run}_4KHZ_R1-{gps}-4096.hdf5
        where parent = (gps // 65536) * 65536 and prefix = det[0] (L for L1, V for V1).
        """
        prefix = det[0]
        parent = (int(f0) // 65536) * 65536
        tag = f"{observing_run}_4KHZ_R1"
        return (
            f"https://gwosc.org/archive/data/{tag}/{parent}/"
            f"{prefix}-{det}_GWOSC_{tag}-{int(f0)}-4096.hdf5"
        )

    def _resolve_and_group(self, segments, det):
        """
        Group mini-segments by GWOSC 4096s file.

        URLs are constructed mathematically — no API calls needed.
        GWOSC O3b 4KHz files follow a deterministic path pattern, so we can
        skip the get_urls() round-trip entirely and go straight to grouping.

        Segments straddling a file boundary get 'first'/'second' partial tasks
        in both adjacent files; the caller concatenates raw pieces before
        downsampling (zero redundant HTTP requests).

        Returns
        -------
        file_tasks : dict
            url → {'same': [(n,b0,b1),...],
                   'first': [(n,b0,b1,f0_b),...],
                   'second': [(n,b0,b1,f0_b),...]}
        n_cross : int
            Number of cross-file segments.
        """
        segs = np.asarray(segments)

        _GWOSC_FILE_DUR = 4096
        by_f0 = collections.defaultdict(lambda: {"same": [], "first": [], "second": []})
        n_cross = 0

        for n, (b0, b1) in enumerate(segs):
            b0, b1 = float(b0), float(b1)
            f0_a = DataReleaseDownloader._gwosc_file_start(b0)
            f0_b = f0_a + _GWOSC_FILE_DUR

            if b1 <= f0_b:
                by_f0[f0_a]["same"].append((n, b0, b1))
            else:
                by_f0[f0_a]["first"].append((n, b0, b1, float(f0_b)))
                by_f0[f0_b]["second"].append((n, b0, b1, float(f0_b)))
                n_cross += 1

        n_files = len(by_f0)
        n_same = sum(len(v["same"]) for v in by_f0.values())
        print(
            f"Grouping {n_files} GWOSC files for {det} "
            f"({n_same} same-file, {n_cross} cross-file) — constructing URLs..."
        )

        # No API calls: build all URLs from GPS times using the known O3b path pattern
        file_tasks = {
            DataReleaseDownloader._construct_file_url(det, f0): tasks
            for f0, tasks in by_f0.items()
        }

        return file_tasks, n_cross

    @staticmethod
    def _download_and_extract_file(url, task_groups, cfg, max_retries=5):
        """
        Download one GWOSC 4096s HDF5 file and extract all requested slices.

        task_groups keys
        ----------------
        'same'   : [(n, b0, b1)]          — segment fully inside this file
        'first'  : [(n, b0, b1, f0_b)]    — segment starts here, ends in next file
        'second' : [(n, b0, b1, f0_b)]    — segment started in previous file, ends here

        Return format
        -------------
        List of tagged tuples:
          ('done',    n, data, meta)          — complete segment, ready to save
          ('first',   n, raw, sr, b0, b1)    — first raw piece; await 'second'
          ('second',  n, raw, sr, b0, b1)    — second raw piece; await 'first'
        On file-level failure:
          ('done',   n, None, {})            for same tasks
          ('first',  n, None, sr, b0, b1)    for first tasks
          ('second', n, None, sr, b0, b1)    for second tasks
        """
        same_tasks   = task_groups.get("same",   [])
        first_tasks  = task_groups.get("first",  [])
        second_tasks = task_groups.get("second", [])

        for attempt in range(max_retries):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
                    tmp_path = tmp.name

                session = DataReleaseDownloader._make_retry_session()
                resp = session.get(url, stream=True, timeout=300)
                resp.raise_for_status()
                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)

                results = []
                with h5py.File(tmp_path, "r") as hf:
                    strain = hf["strain/Strain"][:]
                    t_file = float(hf["meta/GPSstart"][()])
                    duration = float(hf["meta/Duration"][()])
                    sr = len(strain) / duration

                    # --- same-file segments: extract + downsample now ---
                    for n, b0, b1 in same_tasks:
                        try:
                            i0 = max(0, int(round((b0 - t_file) * sr)))
                            i1 = min(len(strain), int(round((b1 - t_file) * sr)))
                            raw = strain[i0:i1].astype(np.float64)
                            data = pycbc_downsample(
                                raw, sr, cfg.sample_rate,
                                cfg.trim, cfg.noise_low_freq_cutoff,
                            )
                            data = data * DYN_RANGE_FAC
                            meta = {
                                "gps_start": b0 + cfg.trim,
                                "gps_end": b1 - cfg.trim,
                                "trim": int(round(cfg.trim * cfg.sample_rate)),
                                "nsamples": len(data),
                                "old_sample_rate": sr,
                                "sample_rate": cfg.sample_rate,
                            }
                            results.append(("done", n, data, meta))
                        except Exception:
                            results.append(("done", n, None, {}))

                    # --- first half of cross-file segment: raw up to file end ---
                    for n, b0, b1, _f0_b in first_tasks:
                        try:
                            i0 = max(0, int(round((b0 - t_file) * sr)))
                            raw = strain[i0:].astype(np.float64)
                            results.append(("first", n, raw, sr, b0, b1))
                        except Exception:
                            results.append(("first", n, None, sr, b0, b1))

                    # --- second half of cross-file segment: raw from file start ---
                    for n, b0, b1, _f0_b in second_tasks:
                        try:
                            i1 = min(len(strain), int(round((b1 - t_file) * sr)))
                            raw = strain[:i1].astype(np.float64)
                            results.append(("second", n, raw, sr, b0, b1))
                        except Exception:
                            results.append(("second", n, None, sr, b0, b1))

                return results

            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    failed = [("done",   n, None, {})          for n, b0, b1        in same_tasks]
                    failed += [("first",  n, None, 0.0, b0, b1) for n, b0, b1, _ in first_tasks]
                    failed += [("second", n, None, 0.0, b0, b1) for n, b0, b1, _ in second_tasks]
                    return failed
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

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

        # Write simple numeric fields directly
        grp.create_dataset("start_time", data=metadata["start_time"])
        grp.create_dataset("end_time", data=metadata["end_time"])

        # Write string fields as standalone UTF-8 datasets
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

        # Store segments data
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

        # logger.info(
        #    f"Fetching GWOSC data for detector {det} ({run}) "
        #    f"using {self.num_workers} worker(s)"
        # )

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
            file_tasks, n_cross = self._resolve_and_group(segments, det)
            print(
                f"  {len(file_tasks)} files to download "
                f"({n_cross} cross-file segments split across adjacent files)"
            )

            # Helper: save one complete (n, data, meta) to bin or HDF5
            def _save_one(n, data, metadata):
                if self.save_bin:
                    if data is None or not isinstance(data, np.ndarray):
                        return
                    nsamp, nbytes, dt, checksum = self._save_segment_bin(bin_fh, data)
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
                    assert nbytes == nsamp * dt.itemsize
                    self._bin_metadata.append(seg_meta)
                    self._bin_sample_cursor += nsamp
                    self._bin_byte_cursor += nbytes
                else:
                    self._save_segment(hf, n, data, det, run, metadata)

            # Accumulator for cross-file partial pieces: n → {kind: (raw, sr, b0, b1)}
            partials = {}

            def _combine_and_save(n, parts, pbar):
                """Concatenate first+second raw pieces, downsample, save."""
                fr, sr, b0, b1 = parts["first"]
                sr_r, b0_r, b1_r = parts["second"][1], parts["second"][2], parts["second"][3]
                sec_raw = parts["second"][0]
                if fr is None or sec_raw is None:
                    pbar.update()
                    return
                raw = np.concatenate([fr, sec_raw])
                try:
                    data = pycbc_downsample(
                        raw, sr, self.dcfg.sample_rate,
                        self.dcfg.trim, self.dcfg.noise_low_freq_cutoff,
                    )
                    data = data * DYN_RANGE_FAC
                    meta = {
                        "gps_start": b0 + self.dcfg.trim,
                        "gps_end": b1 - self.dcfg.trim,
                        "trim": int(round(self.dcfg.trim * self.dcfg.sample_rate)),
                        "nsamples": len(data),
                        "old_sample_rate": sr,
                        "sample_rate": self.dcfg.sample_rate,
                    }
                    _save_one(n, data, meta)
                except Exception:
                    pass
                pbar.update()

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor, \
                    tqdm(total=len(segments)) as pbar:
                pbar.set_description(f"GWOSC-FILE {det}")

                futures = {
                    executor.submit(
                        DataReleaseDownloader._download_and_extract_file,
                        url, tasks, self.dcfg,
                    ): url
                    for url, tasks in file_tasks.items()
                }

                for future in as_completed(futures):
                    for item in future.result():
                        tag = item[0]

                        if tag == "done":
                            _, n, data, meta = item
                            _save_one(n, data, meta)
                            pbar.update()

                        elif tag == "first":
                            _, n, raw, sr, b0, b1 = item
                            partials.setdefault(n, {})["first"] = (raw, sr, b0, b1)
                            if "second" in partials[n]:
                                _combine_and_save(n, partials.pop(n), pbar)

                        elif tag == "second":
                            _, n, raw, sr, b0, b1 = item
                            partials.setdefault(n, {})["second"] = (raw, sr, b0, b1)
                            if "first" in partials[n]:
                                _combine_and_save(n, partials.pop(n), pbar)
        else:
            with tqdm(total=len(segments)) as pbar:
                pbar.set_description("DET_SCIENCE_DATA GWOSC")
                n_written = 0
                for args in tasks:
                    n, data, metadata = DataReleaseDownloader._get_detector_data(*args)
                    if self.save_bin:
                        if data is None or not isinstance(data, np.ndarray):
                            pbar.update()
                            continue
                        nsamp, nbytes, dt, checksum = self._save_segment_bin(bin_fh, data)
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
                        assert nbytes == nsamp * dt.itemsize
                        self._bin_metadata.append(seg_meta)
                        self._bin_sample_cursor += nsamp
                        self._bin_byte_cursor += nbytes
                        n_written += 1
                        if self.proxy_reset_every and n_written % self.proxy_reset_every == 0:
                            logger.info(
                                f"Proxy reset pause: {self.proxy_reset_sleep}s after "
                                f"{n_written} segments ({det})."
                            )
                            time.sleep(self.proxy_reset_sleep)
                    else:
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
            # logger.warning(f"Segments empty for {det} in {run}. Skipping.")
            return None

        # logger.info(f"Downloading segments from {det} for {run}")

        # Validate if segments are okay to download
        final_mask = self._validate_segments(segments, det, run)

        # Store valid boundaries
        record["segments"] = segments[final_mask]
        total_valid_duration = self.durations[final_mask].sum()
        available_valid_duration = self.durations.sum()

        # logger.info(
        #    f"{det} {run}: Available = {available_valid_duration}, "
        #    f"Valid = {total_valid_duration}."
        # )
        # logger.warning("Min duration & data availability might reduce valid duration.")
        return record

    ## --- Main function for end user ---

    def download(self):
        """
        Download all segments from GWOSC and save to disk.

        Iterates over every (detector, observing-run, segments) record in
        :attr:`full_metadata`, calling :meth:`_fetcher` for each.  Records
        with ``None`` value are skipped silently.  Output is written to the
        ``data_release/`` sub-directory under :attr:`save_parent_dir`.
        """
        # Iterate and download records
        for record in self.full_metadata:

            # Cleanup the record
            # record = self._clean_record(record)
            # Ignore if empty record
            if record == None:
                continue

            # Call fetcher to download valid data
            self._fetcher(
                record["segments"],
                record["detector"],
                record["observing_run"],
            )
