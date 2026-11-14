#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GWOSC strain download: segments downsampled and stored as HDF5 or as
one binary file with a JSON segment index, and checksum validation.

Filename        : real_noise.py
Description     : Short description of the file

Created on 2025-11-06 15:00:16

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

import os
import re
import time
import h5py
import json
import hashlib
import warnings
import tempfile
import collections

import numpy as np
import urllib.request
import requests as _requests

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

from gwosc.locate import get_urls as _gwosc_get_urls

# Suppress LAL's stdio redirection warning
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

from gwpy.timeseries import TimeSeries

from pycbc import DYN_RANGE_FAC

from sage.core.hardcode import _GWOSC_ARCHIVE_URL
from sage.core.logger import get_logger
from sage.data.noise.data_quality import zero_spans
from sage.dsp.filters import pycbc_downsample

logger = get_logger(__name__)


@dataclass(frozen=True)
class DownloadConfig:
    """
    Frozen download settings shared with the download workers.

    Attributes
    ----------
    sample_rate : float
        Target sample rate (Hz) after downsampling.
    trim : float
        Seconds trimmed from each edge after downsampling.
    noise_low_freq_cutoff : float
        High-pass frequency (Hz) of the downsampling.
    max_retries : int
        Attempts at a single-segment fetch.
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
        Segment metadata with ``"checksum_algorithm"``,
        ``"byte_offset"``, ``"byte_length"``, ``"checksum"`` and
        ``"segment_index"``.
    strict : bool
        Raise on a mismatch.

    Returns
    -------
    bool
        True if the checksum matches.

    Raises
    ------
    IOError
        If the checksum does not match and ``strict`` is set.
    """
    algo = seg_meta["checksum_algorithm"]

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
    metadata : list of dict
        Segment metadata, as for :func:`validate_segment`.

    Returns
    -------
    list of int
        Indices of the segments that fail the checksum.
    """
    failures = []

    with open(bin_path, "rb") as f:
        for seg in metadata:
            f.seek(seg["byte_offset"])
            raw = f.read(seg["byte_length"])

            h = hashlib.new(seg["checksum_algorithm"])
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
        Hash algorithm name.
    block : int
        Read block size in bytes.

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
    Downloader of GWOSC strain segments to HDF5 or binary files.

    For each (detector, run, segments) record, the segments are
    downsampled, high-passed, trimmed and scaled by PyCBC's
    ``DYN_RANGE_FAC``.

    Parameters
    ----------
    segments_metadata : numpy.ndarray of SEGMENT_DTYPE
        Records from
        :class:`~sage.data.primer.get_segments.TimelineQuery`.
    save_parent_dir : str
        Directory under which the release directory is created.
    noise_low_freq_cutoff : float
        High-pass frequency of the downsampling in Hz.
    minimum_segment_duration : float
        Shortest post-trim segment accepted by
        :meth:`_validate_segments`, in seconds.
    corrupt_trim_length : float
        Seconds trimmed from each edge after downsampling.
    max_download_retries : int
        Attempts at each GWOSC download.
    retry_delay : float
        Seconds between attempts of a single-segment fetch.
    num_workers : int
        Number of download threads; above 1, whole GWOSC files are
        downloaded and sliced.
    proxy_reset_every : int
        With one worker and ``save_bin``, pause after this many
        segments; 0 never pauses.
    proxy_reset_sleep : float
        Length of that pause in seconds.
    make_monolithic_file : bool
        Store each (detector, run) in one file; otherwise one HDF5 file
        per segment.
    save_bin : bool
        Write raw little-endian float32 samples with a JSON segment
        index instead of HDF5.
    sample_rate : float
        Sample rate after downsampling in Hz.
    release_dirname : str
        Name of the release directory under ``save_parent_dir``;
        distinct names keep concurrent downloads apart.
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
        release_dirname: str = "data_release",
    ):

        self.sample_rate = sample_rate
        self.noise_low_freq_cutoff = noise_low_freq_cutoff
        self.trim = corrupt_trim_length
        self.minimum_segment_duration = minimum_segment_duration

        self.max_retries = max_download_retries
        self.delay = retry_delay
        self.num_workers = num_workers
        self.proxy_reset_every = proxy_reset_every
        self.proxy_reset_sleep = proxy_reset_sleep

        self.save_parent_dir = save_parent_dir
        self.release_dirname = release_dirname
        self.monolithic = make_monolithic_file

        self.full_metadata = segments_metadata

        # Frozen copy of the settings for the download workers
        self.dcfg = self._return_download_config()

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
        """
        Open data of a detector over ``[b0, b1]``, with retries.

        Parameters
        ----------
        cfg : DownloadConfig
            Settings; ``max_retries`` attempts, ``delay`` apart.
        det : str
            Detector prefix.
        b0, b1 : float
            GPS bounds.

        Returns
        -------
        tuple
            ``(TimeSeries, True)``, or ``(None, False)`` if every
            attempt failed.
        """
        last_exception = None

        # TODO: generalise sage.core.errors.safe_call to cover this
        for ntry in range(cfg.max_retries):
            try:
                data = TimeSeries.fetch_open_data(det, b0, b1, cache=True)
                return data, True
            except Exception as e:
                last_exception = e
                time.sleep(cfg.delay)

        # Every attempt failed
        if last_exception != None:
            return None, False

    @staticmethod
    def _get_detector_data_unpacked(args):
        """Call :meth:`_get_detector_data` on a tuple of arguments."""
        return DataReleaseDownloader._get_detector_data(*args)

    @staticmethod
    def _get_detector_data(cfg, n, b0, b1, det):
        """
        Download, downsample and scale one segment.

        Parameters
        ----------
        cfg : DownloadConfig
            Download settings.
        n : int
            Segment index, returned as given.
        b0, b1 : float
            GPS bounds before the trim.
        det : str
            Detector prefix.

        Returns
        -------
        tuple
            ``(n, data, metadata)``, or ``(n, None, {})`` if the
            download failed.
        """

        data, fetch_okay = DataReleaseDownloader._fetch_data(cfg, det, b0, b1)
        if not fetch_okay:
            return n, None, {}

        old_sample_rate = 1.0 / data.dt.value
        spans = zero_spans(data.value, b0, old_sample_rate)
        data = pycbc_downsample(
            data.value,
            old_sample_rate,
            cfg.sample_rate,
            cfg.trim,
            cfg.noise_low_freq_cutoff,
        )
        # Stored scaled by DYN_RANGE_FAC, to be divided out on reading
        data = data * DYN_RANGE_FAC

        metadata = {
            "gps_start": b0 + cfg.trim,
            "gps_end": b1 - cfg.trim,
            "trim": int(round(cfg.trim * cfg.sample_rate)),
            "nsamples": len(data),
            "old_sample_rate": old_sample_rate,
            "sample_rate": cfg.sample_rate,
            "zero_spans": spans,
        }
        return n, data, metadata

    def _checksum_array(self, data: np.ndarray, algo="sha256"):
        """
        Checksum of the raw bytes of an array.

        Parameters
        ----------
        data : numpy.ndarray
            Contiguous array.
        algo : str
            Hash algorithm name.

        Returns
        -------
        str
            Hex digest.
        """
        h = hashlib.new(algo)
        h.update(memoryview(data))
        return h.hexdigest()

    def _bin_open(self, filename):
        """Open ``save_dir / filename`` for binary writing."""
        filepath = self.save_dir / filename
        return open(filepath, "wb")

    def _save_segment_bin(self, fh, data):
        """
        Append samples to an open binary file, little-endian.

        Parameters
        ----------
        fh : file object
            File open for binary writing.
        data : numpy.ndarray or None
            Samples.

        Returns
        -------
        tuple
            ``(nsamples, nbytes, dtype, sha256 hex digest)``, or
            ``(0, 0, None)`` if ``data`` is not an array.
        """
        if data is None or not isinstance(data, np.ndarray):
            return 0, 0, None

        dt = np.dtype(data.dtype).newbyteorder("<")
        data = np.ascontiguousarray(data.astype(dt, copy=False))

        raw = data.tobytes(order="C")
        fh.write(raw)
        checksum = hashlib.sha256(raw).hexdigest()

        return data.size, len(raw), dt, checksum

    @staticmethod
    def _file_gps_interval(url):
        """
        GPS interval of a GWOSC file from its URL.

        Parameters
        ----------
        url : str
            URL ending in ``-{gps}-{duration}`` and ``.hdf5``,
            ``.hdf`` or ``.gwf``.

        Returns
        -------
        tuple
            ``(gps_start, gps_end)``, or ``(None, None)`` if the URL
            does not match.
        """
        m = re.search(r"-(\d+)-(\d+)\.(hdf5?|gwf)$", url)
        if m:
            t0 = int(m.group(1))
            return t0, t0 + int(m.group(2))
        return None, None

    @staticmethod
    def _make_retry_session(connect_retries=5, pool_size=32):
        """
        A requests Session that retries connection errors in urllib3.

        Parameters
        ----------
        connect_retries : int
            Retries of each request, with backoff factor 2.
        pool_size : int
            Connection pool size; at least the number of threads.

        Returns
        -------
        requests.Session
            Session with the retrying adapter on http and https.
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
        """
        GPS start of the 4096 s GWOSC file that contains ``t``.

        GWOSC 4096 s files start at multiples of 4096.

        Parameters
        ----------
        t : float
            GPS time.

        Returns
        -------
        int
            File start.
        """
        return (int(t) // 4096) * 4096

    @staticmethod
    def _construct_file_url(det, f0, observing_run="O3b"):
        """
        URL of a GWOSC 4 kHz HDF5 file; the pattern is verified for O3b.

        ``{prefix}-{det}_GWOSC_{run}_4KHZ_R1-{f0}-4096.hdf5`` in the
        archive directory ``{run}_4KHZ_R1/{parent}/``, with ``parent =
        (f0 // 65536) * 65536`` and ``prefix = det[0]``.

        Parameters
        ----------
        det : str
            Detector prefix.
        f0 : int
            GPS start of the file.
        observing_run : str
            Observing run.

        Returns
        -------
        str
            The file URL.
        """
        prefix = det[0]
        parent = (int(f0) // 65536) * 65536
        tag = f"{observing_run}_4KHZ_R1"
        return (
            _GWOSC_ARCHIVE_URL.format(tag=tag, parent=parent)
            + f"{prefix}-{det}_GWOSC_{tag}-{int(f0)}-4096.hdf5"
        )

    def _resolve_and_group(self, segments, det, run):
        """
        Group segments by the GWOSC 4096 s file that holds them.

        URLs are built from the GPS times, without API calls. A
        segment crossing a file boundary gives a ``"first"`` task in
        one file and a ``"second"`` task in the next.

        Parameters
        ----------
        segments : array-like, shape ``(N, 2)``
            GPS ``[start, end]`` of each segment.
        det : str
            Detector prefix.
        run : str
            Observing run.

        Returns
        -------
        file_tasks : dict
            URL -> ``{"same": [(n, b0, b1)], "first": [(n, b0, b1,
            f0_b)], "second": [(n, b0, b1, f0_b)]}``.
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

        file_tasks = {
            DataReleaseDownloader._construct_file_url(det, f0, observing_run=run): tasks
            for f0, tasks in by_f0.items()
        }

        return file_tasks, n_cross

    @staticmethod
    def _download_and_extract_file(url, task_groups, cfg, max_retries=5):
        """
        Download one GWOSC 4096 s HDF5 file and extract its slices.

        A segment inside the file is downsampled here; a cross-file
        segment gives its raw piece in this file.

        Parameters
        ----------
        url : str
            File URL.
        task_groups : dict
            ``"same"``: ``[(n, b0, b1)]`` inside this file; ``"first"``:
            ``[(n, b0, b1, f0_b)]`` starting here; ``"second"``: the
            same, ending here.
        cfg : DownloadConfig
            Download settings.
        max_retries : int
            Attempts at the file download.

        Returns
        -------
        list of tuple
            ``("done", n, data, meta)``,
            ``("first", n, raw, sr, b0, b1)`` or
            ``("second", n, raw, sr, b0, b1)``; ``data`` or ``raw`` is
            None where the slice, or every download attempt, failed.
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

                    # Segments inside this file: extract and downsample
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
                                "zero_spans": zero_spans(raw, b0, sr),
                            }
                            results.append(("done", n, data, meta))
                        except Exception:
                            results.append(("done", n, None, {}))

                    # First piece of a cross-file segment
                    for n, b0, b1, _f0_b in first_tasks:
                        try:
                            i0 = max(0, int(round((b0 - t_file) * sr)))
                            raw = strain[i0:].astype(np.float64)
                            results.append(("first", n, raw, sr, b0, b1))
                        except Exception:
                            results.append(("first", n, None, sr, b0, b1))

                    # Second piece, from the file start
                    for n, b0, b1, _f0_b in second_tasks:
                        try:
                            i1 = min(len(strain), int(round((b1 - t_file) * sr)))
                            raw = strain[:i1].astype(np.float64)
                            results.append(("second", n, raw, sr, b0, b1))
                        except Exception:
                            results.append(("second", n, None, sr, b0, b1))

                return results

            except Exception as exc:
                if attempt < max_retries - 1:
                    # Refused connection: 60 s per attempt, up to 300 s;
                    # other errors: exponential backoff up to 60 s
                    if "Connection refused" in str(exc) or "Failed to establish" in str(exc):
                        wait = min(300, 60 * (attempt + 1))
                    else:
                        wait = min(60, 2 ** attempt)
                    time.sleep(wait)
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
        """
        The download settings as a frozen :class:`DownloadConfig`.

        Returns
        -------
        DownloadConfig
            Settings for the download workers.
        """
        return DownloadConfig(
            sample_rate=self.sample_rate,
            trim=self.trim,
            noise_low_freq_cutoff=self.noise_low_freq_cutoff,
            max_retries=self.max_retries,
            delay=self.delay,
        )

    def _save_metadata(self, hf, group_name, metadata):
        """
        Write segment records to an HDF5 group.

        String fields are written as variable-length UTF-8 datasets, the
        segments of each record as a dataset of their own.

        Parameters
        ----------
        hf : h5py.File
            Open file.
        group_name : str
            Group to write into.
        metadata : numpy.ndarray of SEGMENT_DTYPE
            Records.
        """

        grp = hf.require_group(group_name)

        grp.create_dataset("start_time", data=metadata["start_time"])
        grp.create_dataset("end_time", data=metadata["end_time"])

        dt_str = h5py.string_dtype(encoding="utf-8")

        detector_list = [str(x) for x in metadata["detector"]]
        grp.create_dataset("detector", data=detector_list, dtype=dt_str)

        flag_list = [str(x) for x in metadata["flag"]]
        grp.create_dataset("flag", data=flag_list, dtype=dt_str)

        observing_run_list = [str(x) for x in metadata["observing_run"]]
        grp.create_dataset("observing_run", data=observing_run_list, dtype=dt_str)

        seg_grp = grp.require_group("segments")

        N = len(metadata)
        seg_index = np.zeros(N, dtype="i8")

        for i in range(N):
            # Object field: one array of segments per record
            segs = np.asarray(metadata["segments"][i])
            seg_grp.create_dataset(f"{i:05d}", data=segs)
            seg_index[i] = i

        grp.create_dataset("segments_index", data=seg_index)

    def _savepath_handling(self, dirname):
        """
        Set ``save_dir`` to ``save_parent_dir / dirname`` and create it.

        Parameters
        ----------
        dirname : str
            Name of the directory.
        """
        self.save_dir = Path(self.save_parent_dir) / dirname
        if not os.path.exists(self.save_dir):
            self.save_dir.mkdir(parents=True, exist_ok=False)

    def _h5py_mkfile(self, filename):
        """Open ``save_dir / filename`` as a new HDF5 file."""
        filepath = self.save_dir / filename
        return h5py.File(filepath, "w")

    def _save_segment(self, hf, idx, data, det, run, chunk_metadata):
        """
        Write one segment as a gzip dataset with its metadata.

        Parameters
        ----------
        hf : h5py.File
            Open file.
        idx : int
            Segment index; the dataset name.
        data : numpy.ndarray or None
            Samples; nothing is written for None.
        det : str
            Detector prefix.
        run : str
            Observing run.
        chunk_metadata : dict
            Attributes of the segment.
        """
        if data is None or not isinstance(data, np.ndarray):
            return

        seg_grp = hf.require_group("segments")
        # Zero-padded to five digits
        name = f"{idx:05d}"

        dset = seg_grp.create_dataset(
            name,
            data=data,
            dtype=data.dtype,
            compression="gzip",
            chunks=True,
        )

        for k, v in chunk_metadata.items():
            dset.attrs[k] = v

        dset.attrs["detector"] = det.encode("utf-8")
        dset.attrs["run"] = run.encode("utf-8")
        dset.attrs["noise_low_freq_cutoff"] = self.noise_low_freq_cutoff

    def _fetcher(self, segments, det, run):
        """
        Download one detector's segments and store them.

        Monolithic: one HDF5 file, or one ``.bin`` file with JSON lists
        of the saved and the failed segments. Otherwise one HDF5 file
        per segment.

        Parameters
        ----------
        segments : numpy.ndarray, shape ``(N, 2)``
            GPS ``[start, end]`` of each segment, before the trim.
        det : str
            Detector prefix.
        run : str
            Observing run.
        """

        if self.monolithic and not self.save_bin:
            self._savepath_handling(self.release_dirname)
            hf = self._h5py_mkfile(f"data_{det}_{run}.h5")
            # Records of the full dataset, stored once
            self._save_metadata(hf, "metadata", self.full_metadata)

        elif self.monolithic and self.save_bin:
            self._savepath_handling(self.release_dirname)
            bin_fh = self._bin_open(f"data_{det}_{run}.bin")
            self._bin_metadata = []
            self._failed_metadata = []
            self._bin_sample_cursor = 0
            self._bin_byte_cursor = 0

        else:
            self._savepath_handling(f"data_release_{det}_{run}")

        # One task per segment, for a single worker
        tasks = ((self.dcfg, i, b0, b1, det) for i, (b0, b1) in enumerate(segments))

        def _save_h5(n, data, metadata):
            """Save one segment to the monolithic file, or its own file."""
            if self.monolithic:
                self._save_segment(hf, n, data, det, run, metadata)
                return
            chunk = self._h5py_mkfile(f"data_{det}_{run}_chunk_{n}.hdf")
            self._save_segment(chunk, n, data, det, run, metadata)
            chunk.close()

        if self.num_workers > 1:
            file_tasks, n_cross = self._resolve_and_group(segments, det, run)
            print(
                f"  {len(file_tasks)} files to download "
                f"({n_cross} cross-file segments split across adjacent files)"
            )

            def _save_one(n, data, metadata):
                """Save one complete segment to the output file."""
                if self.save_bin:
                    if data is None or not isinstance(data, np.ndarray):
                        b0, b1 = float(segments[n][0]), float(segments[n][1])
                        self._failed_metadata.append({
                            "segment_index": int(n),
                            "detector": det,
                            "observing_run": run,
                            "gps_start": b0,
                            "gps_end": b1,
                            "reason": "processing_or_download_failed",
                        })
                        return
                    nsamp, nbytes, dt, checksum = self._save_segment_bin(bin_fh, data)
                    seg_meta = {
                        # Position in the saved metadata, not n: a dense
                        # 0..N-1 key whatever the order or failures
                        "segment_index": len(self._bin_metadata),
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
                        "zero_spans": metadata["zero_spans"],
                    }
                    assert nbytes == nsamp * dt.itemsize
                    self._bin_metadata.append(seg_meta)
                    self._bin_sample_cursor += nsamp
                    self._bin_byte_cursor += nbytes
                else:
                    _save_h5(n, data, metadata)

            # Cross-file pieces: n -> {kind: (raw, sr, b0, b1)}
            partials = {}

            def _combine_and_save(n, parts, pbar):
                """Join the two raw pieces, downsample and save."""
                fr, sr, b0, b1 = parts["first"]
                sr_r, b0_r, b1_r = parts["second"][1], parts["second"][2], parts["second"][3]
                sec_raw = parts["second"][0]
                if fr is None or sec_raw is None:
                    self._failed_metadata.append({
                        "segment_index": int(n),
                        "detector": det,
                        "observing_run": run,
                        "gps_start": float(b0),
                        "gps_end": float(b1),
                        "reason": "download_failed_cross_file",
                    })
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
                        "zero_spans": zero_spans(raw, b0, sr),
                    }
                    _save_one(n, data, meta)
                except Exception as exc:
                    self._failed_metadata.append({
                        "segment_index": int(n),
                        "detector": det,
                        "observing_run": run,
                        "gps_start": float(b0),
                        "gps_end": float(b1),
                        "reason": f"processing_failed: {exc}",
                    })
                pbar.update()

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor, \
                    tqdm(total=len(segments)) as pbar:
                pbar.set_description(f"GWOSC-FILE {det}")

                # Bounded batches keep finished results from piling up
                _BATCH = self.num_workers * 4
                file_items = list(file_tasks.items())

                for batch_start in range(0, len(file_items), _BATCH):
                    batch = file_items[batch_start: batch_start + _BATCH]
                    futures = {
                        executor.submit(
                            DataReleaseDownloader._download_and_extract_file,
                            url, tasks, self.dcfg, self.max_retries,
                        ): url
                        for url, tasks in batch
                    }

                    for future in as_completed(futures):
                        items = future.result()
                        # Release the result so its arrays can be freed
                        futures.pop(future, None)

                        for item in items:
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
                            # Positional index, as in _save_one
                            "segment_index": len(self._bin_metadata),
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
                            "zero_spans": metadata["zero_spans"],
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
                        _save_h5(n, data, metadata)
                    pbar.update()

        # Close the monolithic output file
        if self.monolithic and not self.save_bin:
            hf.close()

        elif self.monolithic and self.save_bin:
            bin_fh.close()

            meta_path = self.save_dir / f"data_{det}_{run}_segments.json"
            with open(meta_path, "w") as f:
                json.dump(self._bin_metadata, f, indent=2)

            failed_path = self.save_dir / f"data_{det}_{run}_failed_segments.json"
            with open(failed_path, "w") as f:
                json.dump(self._failed_metadata, f, indent=2)

        elif not self.monolithic:
            metadata_path = Path(self.save_parent_dir) / "full_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(self.full_metadata, f, indent=2)

    def _validate_segments(self, segments, det, run):
        """
        Mask of the segments long enough and available at both edges.

        Sets ``durations`` to the post-trim durations.

        Parameters
        ----------
        segments : numpy.ndarray, shape ``(N, 2)``
            GPS ``[start, end]`` of each segment.
        det : str
            Detector prefix.
        run : str
            Observing run.

        Returns
        -------
        numpy.ndarray, shape ``(N,)``
            True for a valid segment.
        """
        det_start = segments[:, 0]
        det_end = segments[:, 1]
        self.durations = det_end - det_start
        # Durations after the trim of both edges
        self.durations = self.durations - (2.0 * self.trim)

        duration_mask = self.durations >= self.minimum_segment_duration

        # One second at each edge is fetched to check availability
        edge_times = []
        for idx, (start, end) in enumerate(zip(det_start, det_end)):
            edge_times.append((idx, start, start + 1))
            edge_times.append((idx, end - 1, end))

        edge_ok = np.ones(segments.shape[0], dtype=bool)
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

        return duration_mask & edge_ok

    def _clean_record(self, record):
        """
        Keep only the valid segments of a record.

        Parameters
        ----------
        record : numpy.void of SEGMENT_DTYPE
            Record of one detector and run.

        Returns
        -------
        numpy.void or None
            The pruned record, or None if it has no segments.
        """
        run = record["observing_run"]
        det = record["detector"]
        segments = record["segments"]

        if segments.size == 0:
            return None

        final_mask = self._validate_segments(segments, det, run)

        record["segments"] = segments[final_mask]
        total_valid_duration = self.durations[final_mask].sum()
        available_valid_duration = self.durations.sum()

        return record

    def retry_and_append(
        self,
        bin_path,
        seg_json_path,
        segments_to_retry,
        detector: str,
        run: str,
    ):
        """
        Download segments and append them to an existing ``.bin`` file.

        ``seg_json_path`` gains the new entries, and the segments that
        still fail are written to ``*_retry_failed.json`` beside it.

        Parameters
        ----------
        bin_path : Path or str
            Path to the existing monolithic ``.bin`` file.
        seg_json_path : Path or str
            Path to the existing ``*_segments.json`` sidecar.
        segments_to_retry : array-like of shape (N, 2)
            Pre-trim GPS start/end times for each segment to retry.
        detector : str
            Detector name (e.g. ``"H1"``).
        run : str
            Observing run label (e.g. ``"O3b"``).

        Raises
        ------
        ValueError
            If ``segments_to_retry`` is not of shape ``(N, 2)``.
        """
        bin_path = Path(bin_path)
        seg_json_path = Path(seg_json_path)

        with open(seg_json_path) as f:
            existing_meta = json.load(f)

        # Compute append cursors from tail of existing data
        if existing_meta:
            last = max(existing_meta, key=lambda m: m["byte_offset"])
            byte_cursor = last["byte_offset"] + last["byte_length"]
            sample_cursor = last["sample_start_idx"] + last["nsamples"]
            # By count, continuing the dense 0..N-1 sequence
            next_seg_idx = len(existing_meta)
        else:
            byte_cursor = 0
            sample_cursor = 0
            next_seg_idx = 0

        # The cursor follows the actual file size
        actual_size = bin_path.stat().st_size
        if byte_cursor != actual_size:
            print(
                f"WARNING: expected byte_cursor={byte_cursor} but file is {actual_size} bytes. "
                "Adjusting to actual size."
            )
            byte_cursor = actual_size
            if existing_meta:
                dt = np.dtype(existing_meta[0]["dtype"]).newbyteorder(
                    existing_meta[0]["endianness"]
                )
                sample_cursor = actual_size // dt.itemsize

        retry_segs = np.asarray(segments_to_retry, dtype=np.float64)
        if retry_segs.ndim != 2 or retry_segs.shape[1] != 2:
            raise ValueError("segments_to_retry must be shape (N, 2)")

        file_tasks, n_cross = self._resolve_and_group(retry_segs, detector, run)
        print(
            f"  {len(retry_segs)} segments to retry "
            f"({n_cross} cross-file) across {len(file_tasks)} GWOSC files"
        )

        new_meta = []
        still_failed = []
        partials = {}

        with open(bin_path, "ab") as bin_fh:

            def _save_retry(n, data, metadata):
                """Append one segment, or record its failure."""
                nonlocal byte_cursor, sample_cursor, next_seg_idx
                if data is None or not isinstance(data, np.ndarray):
                    still_failed.append({
                        "segment_index": int(n),
                        "detector": detector,
                        "observing_run": run,
                        "gps_start": float(retry_segs[n, 0]),
                        "gps_end": float(retry_segs[n, 1]),
                        "reason": "retry_failed",
                    })
                    return
                dt = np.dtype(data.dtype).newbyteorder("<")
                data = np.ascontiguousarray(data.astype(dt, copy=False))
                raw = data.tobytes(order="C")
                bin_fh.write(raw)
                checksum = hashlib.sha256(raw).hexdigest()
                nsamp = data.size
                nbytes = len(raw)
                new_meta.append({
                    "segment_index": next_seg_idx,
                    "detector": detector,
                    "observing_run": run,
                    "gps_start": metadata["gps_start"],
                    "gps_end": metadata["gps_end"],
                    "sample_rate": metadata["sample_rate"],
                    "nsamples": nsamp,
                    "dtype": dt.name,
                    "endianness": dt.byteorder,
                    "sample_start_idx": sample_cursor,
                    "byte_offset": byte_cursor,
                    "byte_length": nbytes,
                    "checksum": checksum,
                    "checksum_algorithm": "sha256",
                    "dyn_range_fac": float(DYN_RANGE_FAC),
                    "noise_low_freq_cutoff": self.noise_low_freq_cutoff,
                    "zero_spans": metadata["zero_spans"],
                })
                byte_cursor += nbytes
                sample_cursor += nsamp
                next_seg_idx += 1

            def _combine_retry(n, parts, pbar):
                """Join the two raw pieces, downsample and append."""
                fr, sr, b0, b1 = parts["first"]
                sec_raw = parts["second"][0]
                if fr is None or sec_raw is None:
                    still_failed.append({
                        "segment_index": int(n),
                        "detector": detector,
                        "observing_run": run,
                        "gps_start": float(retry_segs[n, 0]),
                        "gps_end": float(retry_segs[n, 1]),
                        "reason": "retry_failed_cross_file",
                    })
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
                        "zero_spans": zero_spans(raw, b0, sr),
                    }
                    _save_retry(n, data, meta)
                except Exception as exc:
                    still_failed.append({
                        "segment_index": int(n),
                        "detector": detector,
                        "observing_run": run,
                        "gps_start": float(retry_segs[n, 0]),
                        "gps_end": float(retry_segs[n, 1]),
                        "reason": f"retry_processing_failed: {exc}",
                    })
                pbar.update()

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor, \
                    tqdm(total=len(retry_segs), desc=f"RETRY {detector}") as pbar:
                _BATCH = self.num_workers * 4
                file_items = list(file_tasks.items())

                for batch_start in range(0, len(file_items), _BATCH):
                    batch = file_items[batch_start: batch_start + _BATCH]
                    futures = {
                        executor.submit(
                            DataReleaseDownloader._download_and_extract_file,
                            url, tasks, self.dcfg, self.max_retries,
                        ): url
                        for url, tasks in batch
                    }
                    for future in as_completed(futures):
                        items = future.result()
                        futures.pop(future, None)
                        for item in items:
                            tag = item[0]
                            if tag == "done":
                                _, n, data, meta = item
                                _save_retry(n, data, meta)
                                pbar.update()
                            elif tag == "first":
                                _, n, raw, sr, b0, b1 = item
                                partials.setdefault(n, {})["first"] = (raw, sr, b0, b1)
                                if "second" in partials[n]:
                                    _combine_retry(n, partials.pop(n), pbar)
                            elif tag == "second":
                                _, n, raw, sr, b0, b1 = item
                                partials.setdefault(n, {})["second"] = (raw, sr, b0, b1)
                                if "first" in partials[n]:
                                    _combine_retry(n, partials.pop(n), pbar)

        # Persist updated metadata
        all_meta = existing_meta + new_meta
        with open(seg_json_path, "w") as f:
            json.dump(all_meta, f, indent=2)

        retry_failed_path = seg_json_path.parent / seg_json_path.name.replace(
            "_segments.json", "_retry_failed.json"
        )
        with open(retry_failed_path, "w") as f:
            json.dump(still_failed, f, indent=2)

        print(
            f"Retry complete: {len(new_meta)} new segments appended, "
            f"{len(still_failed)} still failed → {retry_failed_path.name}"
        )

    def download(self):
        """
        Download every record of ``full_metadata``.

        Each record goes through :meth:`_fetcher`.

        A None record is skipped. Monolithic output goes to
        ``release_dirname`` under ``save_parent_dir``.
        """
        for record in self.full_metadata:

            if record == None:
                continue

            self._fetcher(
                record["segments"],
                record["detector"],
                record["observing_run"],
            )
