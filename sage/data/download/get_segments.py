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
from gwosc.datasets import run_segment

# General
from typing import Union, List, Sequence
from itertools import product

# LOCAL
from sage.core.utils import ensure_sequence
from sage.core.types import SEGMENT_DTYPE


class TimelineQuery:

    ## Static methods from sage.core
    # Make input variables iterable (if not already)
    _ensure_sequence = staticmethod(ensure_sequence)

    def __init__(
        self,
        detector: Union[str, Sequence[str]],
        observing_run: Union[str, Sequence[str], None],
        start: Union[float, int, Sequence[float], Sequence[int], None],
        end: Union[float, int, Sequence[float], Sequence[int], None],
        dq_flag: Union[str, Sequence[str], None],
    ):
        """Retrieve JSON of segment details from GWOSC timeline

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
        self.observing_run = self._ensure_sequence(observing_run)
        self.start = self._ensure_sequence(start)
        self.end = self._ensure_sequence(end)
        self.data_quality_flag = self._ensure_sequence(dq_flag)

        # Detector specification (mandatory)
        self.detector = self._ensure_sequence(detector)

        # Structured array of segments as output
        self.segments = None

    def download_segments(self):
        """_summary_

        Raises:
            ValueError: _description_
        """
        match (self.observing_run, self.start, self.end, self.data_quality_flag):

            # Case 1: Only observing run
            case (runs, None, None, None):
                print(
                    f"Getting all segments for {run} with dq flags matching <DET>_DATA"
                )
                for run, det in product(runs, self.detector):
                    run_start, run_end = run_segment(run)
                    segments = get_segments(f"{det}_DATA", run_start, run_end)
                print(segments)

            # Case 2: Only start & end
            case (None, start, end, None) if start is not None and end is not None:
                print(f"Getting all segments between {start}-{end} for all detectors")
                for det in ["H1", "L1", "V1"]:
                    print(det, len(get_segments(f"{det}_DATA", start, end)))

            # Case 3: Only data-quality flag
            case (None, None, None, flag):
                print(f"Getting all segments for flag {flag}")
                seg = get_segments(flag, 0, 9999999999)
                print(len(seg))

            # Case 4: Flag missing → all detectors
            case (None, None, None, None):
                print("Getting all segments for all detectors")
                for det in ["H1", "L1", "V1"]:
                    print(det, len(get_segments(f"{det}_DATA", 0, 9999999999)))

            # Case 5: Run + start/end → limit by time within run for all dets
            case (run, start, end, None) if start is not None and end is not None:
                print(f"Getting segments for run {run} within {start}-{end}")
                for det in ["H1", "L1", "V1"]:
                    print(det, len(get_segments(f"{det}_DATA", start, end)))

            # Case 6: Fallback (invalid input)
            case _:
                raise ValueError(
                    "Invalid input arguments were provided for TimelineQuery!"
                )


if __name__ == "__main__":
    tq = TimelineQuery("O1", None, None, None)
    tq.download_segments()
