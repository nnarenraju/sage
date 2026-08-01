#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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


# GWOSC API
from gwosc.timeline import get_segments
from gwosc.datasets import run_segment, run_at_gps, find_datasets
from gwosc.api import fetch_allevents_json as _fetch_allevents_json

# General
import time
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from itertools import product
from typing import Union, List, Sequence

# LOCAL
from sage.core.errors import safe_call
from sage.core.utils import to_sequence
from sage.core.typing import SEGMENT_DTYPE
from sage.core.logger import get_logger, setup_logging
from sage.core.hardcode import _DETECTORS, _check_detector_prefixes

logger = get_logger(__name__)
setup_logging("logs")


def get_all_detnames():
    """Return all detector names for reference"""
    return _DETECTORS


def get_all_runnames():
    """Return all run names for reference"""
    return find_datasets(type="run")


def get_all_events():
    """Return all events as JSON from GWOSC"""
    return _fetch_allevents_json(full=True, host="https://gwosc.org")


class TimelineQuery:
    """
    GWOSC segment query engine with multi-case dispatch.

    Accepts any combination of detector names, observing-run labels, GPS
    start/end, and data-quality (DQ) flags and routes to the appropriate GWOSC
    API call via a ``match`` statement (14 supported input combinations).

    Results are stored in :attr:`timeline` as a NumPy structured array with
    dtype :data:`~sage.core.typing.SEGMENT_DTYPE`.  Use :meth:`download_segments`
    to populate the timeline, then optionally :meth:`prune_segments` to remove
    known events or short segments.

    Parameters
    ----------
    detector : str or list[str] or None
        Detector prefix(es) to query (e.g. ``"H1"``, ``["H1", "L1"]``).
    observing_run : str or list[str] or None
        LIGO observing-run label(s) (e.g. ``"O3a"``, ``"O3b"``).
    start, end : float or list[float] or None
        GPS start and end times bounding the query.
    dq_flag : str or list[str] or None
        Data-quality flag(s) to query (e.g. ``"H1_DATA"``).
    auto_clean_empty_timelines : bool
        If ``True``, automatically remove records with empty segment lists
        after :meth:`download_segments` (default ``False``).

    Attributes
    ----------
    timeline : list or np.ndarray (SEGMENT_DTYPE)
        Populated after :meth:`download_segments`; structured array of
        (detector, flag, start_time, end_time, observing_run, segments) rows.
    """

    ## Static methods from sage.core
    # Make input variables iterable (if not already)
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
        """Retrieve segment details from GWOSC

        Args:
            observing_run (Union[str, Sequence[str], None]): {O1, O2, O3, ..., ON}
                label for observing run
            start (Union[float, int, Sequence[float], Sequence[int], None])
                segment start GPS time
            end (Union[float, int, Sequence[float], Sequence[int], None])
                segment end GPS time
            dq_flag (Union[str, Sequence[str], None])
                data quality flag;
                if None <DET>_DATA from all available <DET> returned
        """

        # Parameters for GWOSC query
        self.observing_run = self._to_seq(observing_run)
        self.start = start
        self.end = end

        # if dq flag provided; detector not required
        self.data_quality_flag = self._to_seq(dq_flag)
        # if det provided; only <DET>_data retrieved
        self.detector = self._to_seq(detector)
        # SANITY CHECK: self.detector must be subset of _DETECTORS
        _check_detector_prefixes(self.detector)

        # Structured array of segments as output
        self.timeline = []
        self.auto_clean = auto_clean_empty_timelines

    def extract_det_from_flag(self, flag: str, detectors=_DETECTORS):
        """Return the detector prefix if the flag contains one, else None

        Args:
            flag (str): _description_
            detectors (_type_, optional): _description_. Defaults to _DETECTORS.

        Returns:
            _type_: _description_
        """
        for det in detectors:
            if det in flag:
                return det
        return "NULL"

    def clean_empty_timelines(self):
        """Remove timelines with empty segments"""
        SEG_FIELDS = list(SEGMENT_DTYPE.fields.keys())  # get fieldnames
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

        # Save cleaned list of records
        self.timeline = cleaned
        self._save_as_structured()

    def _save_as_structured(self):
        """Save list of records as structured array"""
        self.timeline = np.array(self.timeline, dtype=SEGMENT_DTYPE)

    def _get_segment_runspan(self, start, end):
        """Get the observing runs spanned by start and end GPS times

        Args:
            start (_type_): _description_
            end (_type_): _description_

        Returns:
            _type_: _description_
        """
        runs = set()

        # If requested segment is very small
        if end - start < 86400:
            return (run_at_gps(start),)

        # Assuming that the time between two runs < 1 day
        logger.warning("Assuming that the time between consecutive runs is > 1 day!")
        for t in np.arange(start, end, step=86400):
            runs.add(run_at_gps(t))

        return tuple(runs)

    def _make_session(self, connect_retries=5):
        """Return a requests.Session with urllib3-level retry for connection errors.

        gwosc's tenacity retry only handles requests.Timeout; RemoteDisconnected
        and other ConnectionError subclasses bypass it and surface immediately.
        A urllib3 Retry on the HTTPAdapter catches those at the socket level,
        before gwosc's tenacity layer even sees the call.
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
        """Safe call get_segments method from GWOSC with retry on failure.

        Retries up to max_retries times with exponential backoff starting at
        base_delay seconds. Returns an empty array only after all retries fail.
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
        """Common function to download segment metadata from GWOSC

        Args:
            runs (_type_): _description_
            start (_type_): _description_
            end (_type_): _description_
            dets (_type_, optional): _description_. Defaults to None.
            flags (_type_, optional): _description_. Defaults to None.
        """
        pass

    def _case_0_handle(self, runs):
        """Handle case 0: Only observing runs provided

        Args:
            runs (_type_): _description_
        """

        logger.info(f"Getting all segments for runs in {runs} and all detectors")
        # Get all available detectors in observing runs
        self._case_4_handle(runs, dets=_DETECTORS)

    def _case_1_handle(self, start, end):
        """Handle case 1: Only start and end provided

        Args:
            start (_type_): _description_
            end (_type_): _description_
        """

        logger.info(f"Getting all segments from <DET>_DATA from all detectors")
        self._case_5_handle(start, end, _DETECTORS)

    def _case_2_handle(self, flags):
        """Handle case 2: Only flags provided

        Args:
            flags (_type_): _description_
        """

        logger.info(f"Getting all segments for flag(s) {flags}")
        logger.warning("Only flags provided; this will likely download a lot of data!")
        start = 0
        end = 999_999_999_999
        self._case_5_handle(start, end, dets=None, flags=flags)

    def _case_3_handle(self, dets):
        """Handle case 3: Only dets provided

        Args:
            dets (_type_): _description_
        """
        logger.info(f"Getting all segments from <DET>_DATA for requested detectors")
        logger.warning("Only dets provided; this will likely download a lot of data!")
        start = 0
        end = 999_999_999_999
        self._case_5_handle(start, end, dets=_DETECTORS, flags=None)

    def _case_4_handle(self, runs, dets=None, flags=None):
        """Handle case 4: Observing run and dets provided

        Args:
            runs (_type_): _description_
            dets (_type_): _description_
        """

        logger.info(
            f"Getting all segments for runs in {runs} "
            "with data quality flags matching <DET>_DATA"
        )
        if flags == None:
            flags = [f"{det}_DATA" for det in dets]
        for run, flag in product(runs, flags):
            run_start, run_end = run_segment(run)
            # Handling timeout errors or non-existent detector
            # If failure, return empty list
            det = self.extract_det_from_flag(flag)
            segments = self._get_segments(flag, run_start, run_end)

            self.timeline.append(
                (det, flag, run_start, run_end, run, segments),
            )

    def _case_5_handle(self, start, end, dets=None, flags=None):
        """Handle case 5: Segment start & end, dets provided

        Args:
            start (_type_): _description_
            end (_type_): _description_
            dets (_type_): _description_
        """

        logger.info(
            f"Getting all segments between {start}-{end} "
            "for all detectors in the correct observing run"
        )

        # What if the start and end span multiple runs?
        runs = self._get_segment_runspan(start, end)
        if flags == None:
            flags = [f"{det}_DATA" for det in dets]

        for run, flag in product(runs, flags):

            # Handling the case of multiple runs
            if len(runs) != 1:
                logger.warning(
                    f"Warning: start/end span multiple runs {runs}. "
                    "Segments will be retrieved per run."
                )

                # 3 conditions: start run, middle run, end run
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
        """Handle case 6: Flags and dets provided (dets ignored)

        Args:
            flags (_type_): _description_
            dets (_type_): _description_
        """

        logger.info(f"Getting all segments for flag {flags} and <DET>")
        logger.warning("Assuming flag provided without <DET> prefix")
        self._case_2_handle(flags)

    def _case_7_handle(self, start, end):
        """Handle case 7: Runs, start and end provided

        Args:
            start (_type_): _description_
            end (_type_): _description_
        """
        logger.warning("Run information ignored")
        logger.info(f"Getting all segments from <DET>_DATA from all detectors")
        self._case_5_handle(start, end, _DETECTORS)

    def _case_8_handle(self, runs, flags):
        """Handle case 8: Runs and flags provided

        Args:
            runs (_type_): _description_
            flags (_type_): _description_
        """
        self._case_4_handle(runs, dets=None, flags=flags)

    def _case_9_handle(self, start, end, flags):
        """Handle case 9: Segment start, end and flags provided

        Args:
            start (_type_): _description_
            end (_type_): _description_
            flags (_type_): _description_
        """
        self._case_5_handle(start, end, flags=flags)

    def _save_and_clean(self):
        """Clean up empty slots and save structured array"""
        # Auto cleanup if requested
        if self.auto_clean:
            self.clean_empty_timelines()
        # Save list of records as structured array
        self._save_as_structured()

    def download_segments(self):
        """Match cases of all possible download scenarios
        We can write this in a super modular way without any redundancy
        but this reduces readability quite a bit.

        Raises:
            ValueError: _description_
        """
        match (
            self.observing_run,
            self.start,
            self.end,
            self.data_quality_flag,
            self.detector,
        ):

            ## --- Single option Cases ---

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

            ## --- Two option Cases ---

            # Case 4: Observing run and dets
            case (runs, None, None, None, dets):
                self._case_4_handle(runs, dets=dets)

            # Case 5: Segment start & end and dets
            case (None, start, end, None, dets) if start and end:
                self._case_5_handle(start, end, dets=dets)

            # Case 6: Data-quality flag and dets (dets ignored)
            case (None, None, None, flags, dets):
                self._case_6_handle(flags)

            # Case 7: Segment start & end and observing run (ignore runs)
            case (runs, start, end, None, None) if start and end:
                self._case_7_handle(start, end)

            # Case 8: Data-quality flag and observing run
            case (runs, None, None, flags, None):
                self._case_8_handle(runs, flags)

            # Case 9: Segment start & end
            case (None, start, end, flags, None) if start and end:
                self._case_9_handle(start, end, flags)

            ## --- Three/four option Cases ---

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

            # Case 14: (runs, start, end, flags, dets) (runs and dets ignored)
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

        # Save structured array of records + clean empty records
        self._save_and_clean()

    def _subtract_windows_from_segments(self, segments, rm_windows):
        """Remove rm_windows from segments

        Args:
            segments (_type_): np.array of shape (N,2)
            rm_windows (_type_): iterable of [start,end] windows to remove

        Returns:
            np.ndarray: new array of shape (M,2)
        """

        # Nothing to do if no segments
        if segments.size == 0:
            return segments

        pruned_segments = []

        for seg_start, seg_end in segments:
            # This will be pruned and split soon
            remaining = [(seg_start, seg_end)]

            # Check and prune rmwindow from segment
            for rm_start, rm_end in rm_windows:
                # TMP store updated segment
                updated = []

                # Look at all three cases of pruning
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
        """Remove all events from GWOSC and return updated segments

        Args:
            rm_length (_type_): _description_
        """
        logger.info(f"Removing all events (obtained from GWOSC) from segments")
        logger.info(f"[event_gps - rm_length, event_gps + rm_length] will be removed")

        allevents = get_all_events()["events"]
        # Iterate through all events, get GPS and get segments to remove
        allevents_gps = np.sort([allevents[key]["GPS"] for key in allevents.keys()])

        # Get all windows that must be removed from segments
        rm_windows = [[inv - rm_length, inv + rm_length] for inv in allevents_gps]

        # Iterate over segments from each record and prune from event list
        prune_timeline = self.timeline.copy()
        for record in prune_timeline:
            segs = record["segments"]
            record["segments"] = self._subtract_windows_from_segments(segs, rm_windows)

        self.timeline = prune_timeline

    def _remove_short_segments(self, rm_min_duration):
        """Remove segments that do not pass the minimum duration threshold

        Args:
            rm_min_duration (_type_): _description_
        """
        logger.info(f"Removing segments below the min duration of {rm_min_duration}")

        # Safe to make copy first
        prune_timeline = self.timeline.copy()

        for i, record in enumerate(prune_timeline):
            segs = record["segments"]

            if segs.size == 0:
                continue  # nothing to prune

            durations = segs[:, 1] - segs[:, 0]
            keep_mask = durations >= rm_min_duration
            prune_timeline[i]["segments"] = segs[keep_mask]

        # Update original timeline with pruned timeline
        self.timeline = prune_timeline

    def prune_segments(
        self,
        rm_short_segments: bool = False,
        rm_min_duration: float = 20,
        rm_allevents: bool = False,
        rm_window_length: float = 30,
    ):
        """Prune segments of events/short-duration-segs and return updated timeline

        Args:
            rmevents (bool, optional): _description_. Defaults to False.
            rmwindow_length (float, optional): _description_. Defaults to 30.
            min_segment_length (float, optional): _description_. Defaults to 20.
        """

        # Any segments that may become too small will be handled next
        # IF the short segment pruning is toggled
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
        Split segments in a structured array into mini-segments for training.

        Each mini-segment:
        - Length mini_segment_length (seconds)
        - Starts exactly 1 sample after previous mini-segment end
        - Last mini-segment kept if >= minimum_segment_duration

        Args:
            struct_array: np.ndarray
                Structured array of detectors and segments (same as your input)
            mini_segment_length: float
                Length of mini-segments in seconds
            minimum_segment_duration: float
                Minimum segment length to keep

        Returns:
            np.ndarray
                Structured array in same format, with 'segments' replaced by mini-segments
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

                # Slide through the segment
                while True:
                    next_end = cursor + mini_segment_length
                    if next_end < seg_end:
                        mini_segments.append([cursor, next_end])
                        # move cursor to 1 sample after previous mini-segment end
                        cursor = (
                            next_end - minimum_segment_duration + (1.0 / sample_rate)
                        )
                    else:
                        # Last partial segment: back-extend to a full mini_segment_length
                        # by moving the start backwards (adding overlap with previous segment).
                        # If the original segment itself is shorter than mini_segment_length,
                        # keep whatever is available from seg_start.
                        remaining = seg_end - cursor
                        if remaining >= minimum_segment_duration:
                            new_start = max(seg_start, seg_end - mini_segment_length)
                            mini_segments.append([new_start, seg_end])
                        break

            # Convert to numpy array
            mini_segments_arr = np.array(mini_segments, dtype=np.float64)

            # Build new record
            new_record = (detector, flag, start_time, end_time, run, mini_segments_arr)
            output_records.append(new_record)

        # Preserve dtype
        self.timeline = np.array(output_records, dtype=prune_timeline.dtype)

    def sanity_check_mini_segments(
        self,
        mini_segment_length=512.0,
        minimum_segment_duration=20.0,
        sample_rate=4096.0,
        verbose=True,
    ):
        """
        Check the integrity of mini-segments in a structured array.

        Args:
            struct_array: np.ndarray
                Structured array with 'segments' field containing mini-segments
            mini_segment_length: float
                Expected mini-segment length (for reference)
            minimum_segment_duration: float
                Minimum segment duration allowed
            verbose: bool
                Whether to print warnings

        Raises:
            ValueError if any check fails
        """
        prune_timeline = self.timeline.copy()
        for rec in prune_timeline:
            detector = rec["detector"]
            original_segments = rec["segments"]
            N = len(original_segments)

            for i, seg in enumerate(original_segments):
                start, end = seg
                duration = end - start

                # Check minimum segment duration
                if duration < minimum_segment_duration:
                    raise ValueError(
                        f"Mini-segment too short: Detector {detector}, segment {i}, duration {duration}"
                    )

                # Check that no segment is longer than expected (optional)
                if duration > mini_segment_length + 1e-6:
                    if verbose:
                        print(
                            f"Warning: Mini-segment longer than intended: Detector {detector}, segment {i}, duration {duration}"
                        )

                # Check gap from previous segment
                if i > 0:
                    prev_end = original_segments[i - 1][1]
                    gap = start - prev_end
                    if gap < -(mini_segment_length + 1.0):
                        # Truly impossible: overlap larger than one full mini-segment
                        raise ValueError(
                            f"Mini-segment {i} overlaps previous segment for detector {detector}"
                        )
                    elif gap < -(minimum_segment_duration - (1.0 / sample_rate)) - 1e-6:
                        # Back-extended last segment — expected, just warn
                        if verbose:
                            print(
                                f"Warning: Back-extended overlap at segment {i} for {detector}: {-gap:.2f}s"
                            )
                    elif gap > 1.0:
                        if verbose:
                            print(
                                f"Warning: Large gap between segments: Detector {detector}, segment {i}, gap {gap} s"
                            )
