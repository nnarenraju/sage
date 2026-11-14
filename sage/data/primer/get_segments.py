#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GWOSC segment queries: run, detector and event lists, and a timeline
of data segments per detector, pruned and split into mini-segments.

Filename        : get_segments.py
Description     : Short description of the file

Created on 2025-11-06 15:06:12

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, Sage
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = inUsage


GitHub Repository: NULL

Documentation: NULL

"""

from gwosc.timeline import get_segments
from gwosc.datasets import run_segment, run_at_gps, find_datasets
from gwosc.api import fetch_allevents_json as _fetch_allevents_json

import time
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from itertools import product
from typing import Union, List, Sequence

from sage.core.hardcode import _GWOSC_HOST
from sage.core.errors import safe_call
from sage.core.utils import to_sequence
from sage.core.typing import SEGMENT_DTYPE
from sage.core.logger import get_logger
from sage.core.hardcode import _DETECTORS, _check_detector_prefixes

# Logging is configured by the caller (sage.core.logger.setup_logging)
logger = get_logger(__name__)


def get_all_detnames():
    """
    Detector prefixes known to Sage.

    Returns
    -------
    sequence of str
        ``_DETECTORS``.
    """
    return _DETECTORS


def get_all_runnames():
    """
    Observing-run names on GWOSC.

    Returns
    -------
    list of str
        ``gwosc.datasets.find_datasets(type="run")``.
    """
    return find_datasets(type="run")


def get_all_events():
    """
    GWOSC's full event list.

    Returns
    -------
    dict
        The allevents JSON, with every event under ``"events"``.
    """
    return _fetch_allevents_json(full=True, host=_GWOSC_HOST)


class TimelineQuery:
    """
    GWOSC data segments for given runs, times, flags and detectors.

    :meth:`download_segments` fills :attr:`timeline`;
    :meth:`prune_segments` and :meth:`split_into_mini_segments` then
    remove events or short segments and split the rest.

    Parameters
    ----------
    detector : str or list of str or None
        Detector prefixes (e.g. ``"H1"``, ``["H1", "L1"]``); None queries
        every detector Sage knows.
    observing_run : str or list of str or None
        Observing-run labels (e.g. ``"O3a"``, ``"O3b"``).
    start, end : float or None
        GPS start and end times bounding the query.
    dq_flag : str or list of str or None
        Data-quality flags to query (e.g. ``"H1_DATA"``).
    auto_clean_empty_timelines : bool
        Remove records without segments after
        :meth:`download_segments`.

    Attributes
    ----------
    timeline : list or numpy.ndarray of SEGMENT_DTYPE
        Records of (detector, flag, start_time, end_time, observing_run,
        segments); a structured array after :meth:`download_segments`.

    Raises
    ------
    AssertionError
        If a detector prefix is unknown.
    """

    # Wraps a scalar input in a tuple
    _to_seq = staticmethod(to_sequence)

    def __init__(
        self,
        detector: Union[str, Sequence[str], None] = None,
        observing_run: Union[str, Sequence[str], None] = None,
        start: Union[float, int, Sequence[float], Sequence[int], None] = None,
        end: Union[float, int, Sequence[float], Sequence[int], None] = None,
        dq_flag: Union[str, Sequence[str], None] = None,
        auto_clean_empty_timelines: bool = False,
    ):
        self.observing_run = self._to_seq(observing_run)
        self.start = start
        self.end = end

        self.data_quality_flag = self._to_seq(dq_flag)
        self.detector = self._to_seq(detector)
        if self.detector is not None:
            _check_detector_prefixes(self.detector)

        self.timeline = []
        self.auto_clean = auto_clean_empty_timelines

    def extract_det_from_flag(self, flag: str, detectors=_DETECTORS):
        """
        Detector prefix contained in a data-quality flag.

        Parameters
        ----------
        flag : str
            Data-quality flag, e.g. ``"H1_DATA"``.
        detectors : sequence of str
            Prefixes to look for.

        Returns
        -------
        str
            The first prefix found in ``flag``, else ``"NULL"``.
        """
        for det in detectors:
            if det in flag:
                return det
        return "NULL"

    def clean_empty_timelines(self):
        """Remove the records without segments from ``timeline``."""
        SEG_FIELDS = list(SEGMENT_DTYPE.fields.keys())
        segments_idx = SEG_FIELDS.index("segments")
        det_idx = SEG_FIELDS.index("detector")
        run_idx = SEG_FIELDS.index("observing_run")

        cleaned = []
        for row in self.timeline:
            segments = row[segments_idx]
            if isinstance(segments, (list, np.ndarray)) and len(segments) == 0:
                logger.debug(
                    f"Skipping empty segment row: det={row[det_idx]}, "
                    f"run={row[run_idx]}"
                )
                continue
            cleaned.append(row)

        self.timeline = cleaned
        self._save_as_structured()

    def _save_as_structured(self):
        """Convert ``timeline`` to a SEGMENT_DTYPE structured array."""
        self.timeline = np.array(self.timeline, dtype=SEGMENT_DTYPE)

    def _get_segment_runspan(self, start, end):
        """
        Observing runs spanned by a GPS interval.

        A span under a day gives the run at ``start``; a longer span is
        sampled once a day.

        Parameters
        ----------
        start, end : float
            GPS bounds of the interval.

        Returns
        -------
        tuple of str
            The runs found.
        """
        runs = set()

        if end - start < 86400:
            return (run_at_gps(start),)

        # A run shorter than a day can fall between two samples
        logger.warning("Assuming that the time between consecutive runs is > 1 day!")
        for t in np.arange(start, end, step=86400):
            runs.add(run_at_gps(t))

        return tuple(runs)

    def _make_session(self, connect_retries=5):
        """
        A requests Session that retries connection errors in urllib3.

        gwosc's own retry handles only ``requests.Timeout``.

        Parameters
        ----------
        connect_retries : int
            Retries of each request, with backoff factor 2.

        Returns
        -------
        requests.Session
            Session with the retrying adapter on http and https.
        """
        session = requests.Session()
        retry = Retry(
            total=connect_retries,
            connect=connect_retries,
            read=connect_retries,
            backoff_factor=2.0,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _get_segments(self, flag, start, end, max_retries=10, base_delay=15):
        """
        GWOSC segments of a flag, retried with exponential backoff.

        Parameters
        ----------
        flag : str
            Data-quality flag.
        start, end : float
            GPS bounds of the query.
        max_retries : int
            Number of attempts.
        base_delay : float
            Wait in seconds after the first failure, doubled each time.

        Returns
        -------
        numpy.ndarray
            Segments as ``(N, 2)`` GPS pairs; empty if every attempt
            failed.
        """
        for attempt in range(max_retries):
            result = safe_call(
                get_segments, flag, start, end,
                fallback_return=None,
            )
            if result is not None:
                return np.array(result)
            if attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt)
                logger.warning(
                    f"_get_segments failed for {flag} (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
        logger.error(f"_get_segments exhausted all {max_retries} retries for {flag}. Returning empty.")
        return np.array([])

    def _download_segment_metadata(
        self, runs=None, start=None, end=None, dets=None, flags=None
    ):
        """
        Unimplemented; does nothing.

        Parameters
        ----------
        runs, start, end, dets, flags
            Unused.
        """
        pass

    def _case_0_handle(self, runs):
        """
        Case 0, runs only: <DET>_DATA of every detector in the runs.

        Parameters
        ----------
        runs : sequence of str
            Observing runs.
        """

        logger.info(f"Getting all segments for runs in {runs} and all detectors")
        self._case_4_handle(runs, dets=_DETECTORS)

    def _case_1_handle(self, start, end):
        """
        Case 1, times only: <DET>_DATA of every detector.

        Parameters
        ----------
        start, end : float
            GPS bounds of the query.
        """

        logger.info(f"Getting all segments from <DET>_DATA from all detectors")
        self._case_5_handle(start, end, _DETECTORS)

    def _case_2_handle(self, flags):
        """
        Case 2, flags only: the flags over all GPS times.

        Parameters
        ----------
        flags : sequence of str
            Data-quality flags.
        """

        logger.info(f"Getting all segments for flag(s) {flags}")
        logger.warning("Only flags provided; this will likely download a lot of data!")
        start = 0
        end = 999_999_999_999
        self._case_5_handle(start, end, dets=None, flags=flags)

    def _case_3_handle(self, dets):
        """
        Case 3, detectors only: <DET>_DATA over all GPS times.

        Parameters
        ----------
        dets : sequence of str
            Detector prefixes.
        """
        logger.info(f"Getting all segments from <DET>_DATA for requested detectors")
        logger.warning("Only dets provided; this will likely download a lot of data!")
        start = 0
        end = 999_999_999_999
        self._case_5_handle(start, end, dets=dets, flags=None)

    def _case_4_handle(self, runs, dets=None, flags=None):
        """
        Case 4, runs and detectors: each flag over each run's span.

        Parameters
        ----------
        runs : sequence of str
            Observing runs.
        dets : sequence of str or None
            Detector prefixes; their <DET>_DATA flags are used when
            ``flags`` is None.
        flags : sequence of str or None
            Data-quality flags.
        """

        logger.info(
            f"Getting all segments for runs in {runs} "
            "with data quality flags matching <DET>_DATA"
        )
        if flags == None:
            flags = [f"{det}_DATA" for det in dets]
        for run, flag in product(runs, flags):
            run_start, run_end = run_segment(run)
            # An empty array if every query attempt fails
            det = self.extract_det_from_flag(flag)
            segments = self._get_segments(flag, run_start, run_end)

            self.timeline.append(
                (det, flag, run_start, run_end, run, segments),
            )

    def _case_5_handle(self, start, end, dets=None, flags=None):
        """
        Case 5, times and detectors: each flag within each spanned run.

        Parameters
        ----------
        start, end : float
            GPS bounds of the query.
        dets : sequence of str or None
            Detector prefixes; their <DET>_DATA flags are used when
            ``flags`` is None.
        flags : sequence of str or None
            Data-quality flags.
        """

        logger.info(
            f"Getting all segments between {start}-{end} "
            "for all detectors in the correct observing run"
        )

        runs = self._get_segment_runspan(start, end)
        if flags == None:
            flags = [f"{det}_DATA" for det in dets]

        for run, flag in product(runs, flags):

            if len(runs) != 1:
                logger.warning(
                    f"Warning: start/end span multiple runs {runs}. "
                    "Segments will be retrieved per run."
                )

                # First, last or middle run of the span
                if run == runs[0]:
                    run_start = start
                    run_end = run_segment(run)[1]
                elif run == runs[-1]:
                    run_start = run_segment(run)[0]
                    run_end = end
                else:
                    run_start, run_end = run_segment(run)
            else:
                run_start, run_end = start, end

            det = self.extract_det_from_flag(flag)
            segments = self._get_segments(flag, run_start, run_end)

            self.timeline.append(
                (det, flag, run_start, run_end, run, segments),
            )

    def _case_6_handle(self, flags):
        """
        Case 6, flags and detectors: case 2, the detectors ignored.

        Parameters
        ----------
        flags : sequence of str
            Data-quality flags.
        """

        logger.info(f"Getting all segments for flag {flags} and <DET>")
        logger.warning("Assuming flag provided without <DET> prefix")
        self._case_2_handle(flags)

    def _case_7_handle(self, start, end):
        """
        Case 7, runs and times: as case 1; the runs are ignored.

        Parameters
        ----------
        start, end : float
            GPS bounds of the query.
        """
        logger.warning("Run information ignored")
        logger.info(f"Getting all segments from <DET>_DATA from all detectors")
        self._case_5_handle(start, end, _DETECTORS)

    def _case_8_handle(self, runs, flags):
        """
        Case 8, runs and flags: each flag over each run's span.

        Parameters
        ----------
        runs : sequence of str
            Observing runs.
        flags : sequence of str
            Data-quality flags.
        """
        self._case_4_handle(runs, dets=None, flags=flags)

    def _case_9_handle(self, start, end, flags):
        """
        Case 9, times and flags: each flag within each spanned run.

        Parameters
        ----------
        start, end : float
            GPS bounds of the query.
        flags : sequence of str
            Data-quality flags.
        """
        self._case_5_handle(start, end, flags=flags)

    def _save_and_clean(self):
        """Remove empty records if requested, then store as an array."""
        if self.auto_clean:
            self.clean_empty_timelines()
        self._save_as_structured()

    def download_segments(self):
        """
        Query GWOSC for the combination of inputs into ``timeline``.

        Raises
        ------
        ValueError
            If the combination of inputs matches no case.
        """
        match (
            self.observing_run,
            self.start,
            self.end,
            self.data_quality_flag,
            self.detector,
        ):

            # Single-input cases

            # Case 0: Only observing run
            case (runs, None, None, None, None):
                self._case_0_handle(runs)

            # Case 1: Only segment start & end
            case (None, start, end, None, None) if start and end:
                self._case_1_handle(start, end)

            # Case 2: Only data-quality flag
            case (None, None, None, flags, None):
                self._case_2_handle(flags)

            # Case 3: Only detectors
            case (None, None, None, None, dets):
                self._case_3_handle(dets)

            # Two-input cases

            # Case 4: Observing run and dets
            case (runs, None, None, None, dets):
                self._case_4_handle(runs, dets=dets)

            # Case 5: Segment start & end and dets
            case (None, start, end, None, dets) if start and end:
                self._case_5_handle(start, end, dets=dets)

            # Case 6: Data-quality flag and dets (dets ignored)
            case (None, None, None, flags, dets):
                self._case_6_handle(flags)

            # Case 7: Segment start & end and run (runs ignored)
            case (runs, start, end, None, None) if start and end:
                self._case_7_handle(start, end)

            # Case 8: Data-quality flag and observing run
            case (runs, None, None, flags, None):
                self._case_8_handle(runs, flags)

            # Case 9: Segment start & end and data-quality flag
            case (None, start, end, flags, None) if start and end:
                self._case_9_handle(start, end, flags)

            # Three- and four-input cases

            # Case 10: (runs, start, end, None, dets) (runs ignored)
            case (runs, start, end, None, dets) if start and end:
                self._case_5_handle(start, end, dets=dets)

            # Case 11: (runs, None, None, flags, dets) (dets ignored)
            case (runs, None, None, flags, dets):
                self._case_8_handle(runs, flags)

            # Case 12: (None, start, end, flags, dets) (dets ignored)
            case (None, start, end, flags, dets):
                self._case_9_handle(start, end, flags)

            # Case 13: (runs, start, end, flags, None) (runs ignored)
            case (runs, start, end, flags, None):
                self._case_9_handle(start, end, flags)

            # Case 14: all inputs (runs and dets ignored)
            case (runs, start, end, flags, dets):
                self._case_9_handle(start, end, flags)

            # Case _: Fallback (invalid input)
            case _:
                error = (
                    "Insufficient/Invalid input arguments "
                    "were provided for TimelineQuery!"
                )
                logger.critical(error)
                raise ValueError(error)

        self._save_and_clean()

    def _subtract_windows_from_segments(self, segments, rm_windows):
        """
        Segments with the given windows cut out.

        Parameters
        ----------
        segments : numpy.ndarray, shape ``(N, 2)``
            GPS ``[start, end]`` segments.
        rm_windows : iterable
            GPS ``[start, end]`` windows to remove.

        Returns
        -------
        numpy.ndarray, shape ``(M, 2)``
            The remaining pieces.
        """

        if segments.size == 0:
            return segments

        pruned_segments = []

        for seg_start, seg_end in segments:
            remaining = [(seg_start, seg_end)]

            for rm_start, rm_end in rm_windows:
                updated = []

                for pruned_start, pruned_end in remaining:

                    # 1. No overlap
                    if pruned_end <= rm_start or pruned_start >= rm_end:
                        updated.append((pruned_start, pruned_end))
                        continue

                    # 2a. Partial or full overlap (left remainder)
                    if pruned_start < rm_start:
                        updated.append((pruned_start, min(pruned_end, rm_start)))

                    # 2b. Partial or full overlap (right remainder)
                    if pruned_end > rm_end:
                        updated.append((max(pruned_start, rm_end), pruned_end))

                remaining = updated

            pruned_segments.extend(remaining)

        return np.array(pruned_segments, dtype=float).reshape(-1, 2)

    def _remove_allevents(
        self,
        rm_length,
    ):
        """
        Cut ``[gps - rm_length, gps + rm_length]`` around each event.

        The events are GWOSC's full event list.

        Parameters
        ----------
        rm_length : float
            Half-width of the cut window in seconds.
        """
        logger.info(f"Removing all events (obtained from GWOSC) from segments")
        logger.info(f"[event_gps - rm_length, event_gps + rm_length] will be removed")

        allevents = get_all_events()["events"]
        allevents_gps = np.sort([allevents[key]["GPS"] for key in allevents.keys()])

        rm_windows = [[inv - rm_length, inv + rm_length] for inv in allevents_gps]

        prune_timeline = self.timeline.copy()
        for record in prune_timeline:
            segs = record["segments"]
            record["segments"] = self._subtract_windows_from_segments(segs, rm_windows)

        self.timeline = prune_timeline

    def _remove_short_segments(self, rm_min_duration):
        """
        Remove segments shorter than ``rm_min_duration``.

        Parameters
        ----------
        rm_min_duration : float
            Shortest segment kept, in seconds.
        """
        logger.info(f"Removing segments below the min duration of {rm_min_duration}")

        prune_timeline = self.timeline.copy()

        for i, record in enumerate(prune_timeline):
            segs = record["segments"]

            if segs.size == 0:
                continue

            durations = segs[:, 1] - segs[:, 0]
            keep_mask = durations >= rm_min_duration
            prune_timeline[i]["segments"] = segs[keep_mask]

        self.timeline = prune_timeline

    def prune_segments(
        self,
        rm_short_segments: bool = False,
        rm_min_duration: float = 20,
        rm_allevents: bool = False,
        rm_window_length: float = 30,
    ):
        """
        Cut the GWOSC events out of ``timeline``, then short segments.

        Parameters
        ----------
        rm_short_segments : bool
            Remove segments shorter than ``rm_min_duration``.
        rm_min_duration : float
            Shortest segment kept, in seconds.
        rm_allevents : bool
            Cut a window around every GWOSC event.
        rm_window_length : float
            Half-width of the event window in seconds.
        """

        # Events first, so the pieces they leave are length-checked
        if rm_allevents:
            self._remove_allevents(rm_window_length)

        if rm_short_segments:
            self._remove_short_segments(rm_min_duration)

    def split_into_mini_segments(
        self,
        mini_segment_length=512.0,
        minimum_segment_duration=20.0,
        sample_rate=4096.0,
    ):
        """
        Split every segment of ``timeline`` into mini-segments.

        Neighbours overlap by ``minimum_segment_duration`` less a
        sample. A remainder that long gives a last mini-segment,
        back-extended to full length within the segment.

        Parameters
        ----------
        mini_segment_length : float
            Length of a mini-segment in seconds.
        minimum_segment_duration : float
            Overlap plus one sample, and the shortest remainder kept.
        sample_rate : float
            Sample rate in Hz, for the one-sample offset.
        """
        output_records = []
        prune_timeline = self.timeline.copy()
        for record in prune_timeline:
            detector = record["detector"]
            flag = record["flag"]
            start_time = record["start_time"]
            end_time = record["end_time"]
            run = record["observing_run"]
            original_segments = record["segments"]  # shape (N,2)

            mini_segments = []

            for seg in original_segments:
                seg_start, seg_end = seg
                cursor = seg_start

                while True:
                    next_end = cursor + mini_segment_length
                    if next_end < seg_end:
                        mini_segments.append([cursor, next_end])
                        # Overlap: minimum duration less a sample
                        cursor = (
                            next_end - minimum_segment_duration + (1.0 / sample_rate)
                        )
                    else:
                        # Last mini-segment, back-extended to full
                        # length but not before seg_start
                        remaining = seg_end - cursor
                        if remaining >= minimum_segment_duration:
                            new_start = max(seg_start, seg_end - mini_segment_length)
                            mini_segments.append([new_start, seg_end])
                        break

            mini_segments_arr = np.array(mini_segments, dtype=np.float64)

            new_record = (detector, flag, start_time, end_time, run, mini_segments_arr)
            output_records.append(new_record)

        self.timeline = np.array(output_records, dtype=prune_timeline.dtype)

    def sanity_check_mini_segments(
        self,
        mini_segment_length=512.0,
        minimum_segment_duration=20.0,
        sample_rate=4096.0,
        verbose=True,
    ):
        """
        Check the mini-segments of ``timeline``.

        Prints a warning for a mini-segment longer than intended, a
        back-extended overlap or a gap above 1 s.

        Parameters
        ----------
        mini_segment_length : float
            Intended mini-segment length in seconds.
        minimum_segment_duration : float
            Shortest mini-segment allowed, in seconds.
        sample_rate : float
            Sample rate in Hz.
        verbose : bool
            Print the warnings.

        Raises
        ------
        ValueError
            If a mini-segment is too short, or overlaps the previous one
            by more than ``mini_segment_length + 1`` s.
        """
        prune_timeline = self.timeline.copy()
        for rec in prune_timeline:
            detector = rec["detector"]
            original_segments = rec["segments"]
            N = len(original_segments)

            for i, seg in enumerate(original_segments):
                start, end = seg
                duration = end - start

                if duration < minimum_segment_duration:
                    raise ValueError(
                        f"Mini-segment too short: Detector {detector}, segment {i}, duration {duration}"
                    )

                # Longer than intended: a warning only
                if duration > mini_segment_length + 1e-6:
                    if verbose:
                        print(
                            f"Warning: Mini-segment longer than intended: Detector {detector}, segment {i}, duration {duration}"
                        )

                if i > 0:
                    prev_end = original_segments[i - 1][1]
                    gap = start - prev_end
                    if gap < -(mini_segment_length + 1.0):
                        # Overlap longer than a full mini-segment
                        raise ValueError(
                            f"Mini-segment {i} overlaps previous segment for detector {detector}"
                        )
                    elif gap < -(minimum_segment_duration - (1.0 / sample_rate)) - 1e-6:
                        # Back-extended last mini-segment
                        if verbose:
                            print(
                                f"Warning: Back-extended overlap at segment {i} for {detector}: {-gap:.2f}s"
                            )
                    elif gap > 1.0:
                        if verbose:
                            print(
                                f"Warning: Large gap between segments: Detector {detector}, segment {i}, gap {gap} s"
                            )
