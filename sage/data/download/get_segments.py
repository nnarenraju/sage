#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : get_segments.py
Description     : Short description of the file

Created on 2025-11-06 15:06:12

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


# GWOSC API
from gwosc.timeline import get_segments
from gwosc.datasets import run_segment, run_at_gps, find_datasets

# General
import numpy as np

from itertools import product
from typing import Union, List, Sequence

# LOCAL
from sage.core.utils import to_sequence
from sage.core.types import SEGMENT_DTYPE
from sage.core.logger import get_logger
from sage.core.hardcode import _DETECTORS, _check_detector_prefixes

logger = get_logger(__name__)


class TimelineQuery:

    ## Static methods from sage.core
    # Make input variables iterable (if not already)
    _to_seq = staticmethod(to_sequence)

    def __init__(
        self,
        detector: Union[str, Sequence[str], None],
        observing_run: Union[str, Sequence[str], None],
        start: Union[float, int, Sequence[float], Sequence[int], None],
        end: Union[float, int, Sequence[float], Sequence[int], None],
        dq_flag: Union[str, Sequence[str], None],
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
        self.start = self._to_seq(start)
        self.end = self._to_seq(end)

        # if dq flag provided; detector not required
        self.data_quality_flag = self._to_seq(dq_flag)
        # if det provided; only <DET>_data retrieved
        self.detector = self._to_seq(detector)
        # SANITY CHECK: self.detector must be subset of _DETECTORS
        _check_detector_prefixes(self.detector)

        # Structured array of segments as output
        self.timeline = []

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

    def _case_1_handle(self, runs, dets):
        """Handle case 1: Observing run and dets provided

        Args:
            runs (_type_): _description_
            dets (_type_): _description_
        """

        logger.info(
            f"Getting all segments for runs in {runs} "
            "with data quality flags matching <DET>_DATA"
        )
        for run, det in product(runs, dets):
            run_start, run_end = run_segment(run)
            segments = np.array(get_segments(f"{det}_DATA", run_start, run_end))
            self.timeline.append(
                np.array(
                    [(det, run_start, run_end, run, segments)],
                    dtype=SEGMENT_DTYPE,
                )
            )

    def _case_2_handle(self, start, end, dets):
        """Handle case 2: Segment start & end, dets provided

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

        for run, det in product(runs, dets):

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

            segments = np.array(get_segments(f"{det}_DATA", run_start, run_end))
            # Get run from segment time
            # Handle case where start/end span multiple runs

            self.timeline.append(
                np.array(
                    [(det, run_start, run_end, run, segments)],
                    dtype=SEGMENT_DTYPE,
                )
            )

    def download_segments(self):
        """_summary_

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
                logger.info(f"Getting all segments for runs in {runs}")

            # Case 1: Only segment start & end
            case (None, start, end, None, None) if start and end:
                logger.info(f"Getting all segments for flag {flag}")
                logger.warning("Assuming <DET> prefix provided along with flag")

            # Case 2: Only data-quality flag
            case (None, None, None, flags, None):
                logger.info(f"Getting all segments for flag {flag}")
                logger.warning("Assuming <DET> prefix provided along with flag")

            # Case 3: Only detectors
            case (None, None, None, None, dets):
                logger.info(f"Getting all segments for flag {flag}")
                logger.warning("Assuming <DET> prefix provided along with flag")

            ## --- Two option Cases ---

            # Case 4: Observing run and dets
            case (runs, None, None, None, dets):
                self._case_1_handle(runs, dets)

            # Case 5: Segment start & end and dets
            case (None, start, end, None, dets) if start and end:
                self._case_2_handle(start, end)

            # Case 6: Data-quality flag and dets
            case (None, None, None, flags, dets):
                logger.info(f"Getting all segments for flag {flag} and <DET>")
                logger.warning("Assuming flag provided without <DET> prefix")

            # Case 7: Segment start & end and observing run
            case (runs, start, end, None, None) if start and end:
                pass

            # Case 8: Data-quality flag and observing run
            case (runs, None, None, flags, None):
                pass

            # Case 9: Segment start & end
            case (None, start, end, flags, None) if start and end:
                pass

            ## --- Three/four option Cases (ignoring conditions) ---
            # Case 10: (runs, start, end, None, dets) (runs ignored)
            # Case 11: (runs, None, None, flags, dets) (dets ignored)
            # Case 12: (None, start, end, flags, dets) (dets ignored)
            # Case 13: (runs, start, end, flags, None) (runs ignored)
            # Case 14: (runs, start, end, flags, dets) (runs and dets ignored)

            # Case _: Fallback (invalid input)
            case _:
                error = (
                    "Insufficient/Invalid input arguments "
                    "were provided for TimelineQuery!"
                )
                logger.critical(error)
                raise ValueError(error)


if __name__ == "__main__":
    tq = TimelineQuery(["H1", "L1"], "O1", None, None, None)
    tq.download_segments()
