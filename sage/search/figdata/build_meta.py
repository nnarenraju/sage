#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : build_meta.py
Description   : Figure data describing the search itself.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress
"""

from pathlib import Path
from typing import Dict, Optional

from sage.search.figdata.product import FigData


def livetime_and_duty_cycle(spec) -> FigData:
    """
    Analysed livetime and where the rest of the run went, per arm.

    ``pycbc_page_segplot``. Every rate the search reports divides by the analysed time,
    so the decomposition is shown rather than asserted: how much each detector observed,
    how much was coincident, how much of that could host a whole window, and what the
    remainder was lost to.

    Read from the recorded stage reports rather than recomputed. ``segments`` measured
    these numbers on the segment lists the campaign actually ran on, and a figure that
    re-derived them could disagree with the livetime the rates were divided by -- which
    is the one disagreement that would not show up as an error anywhere.
    """
    import numpy as np

    from sage.search.manifest import RunManifest
    from sage.search.stages import manifest_path

    summary = RunManifest(path=manifest_path(spec)).summary()
    stages = summary.get("stages", {})
    segments = dict(stages.get("segments", {}) or {})
    grid = dict(stages.get("grid", {}) or {})
    if not segments:
        raise KeyError(
            "the campaign has no recorded 'segments' report, so the livetime "
            "decomposition it measured is not available; run the segments stage first"
        )

    coincident = float(segments.get("coincident_livetime_s", 0.0))
    analysed = float(grid.get("analysed_livetime_s", segments.get("hosted_s", 0.0)))
    # Per detector, as the segments stage recorded them: `union_livetime_s_<detector>`.
    observing = {
        key[len("union_livetime_s_"):]: float(value)
        for key, value in segments.items()
        if key.startswith("union_livetime_s_")
    }
    detectors = tuple(spec.data.detectors)
    observing_s = np.array(
        [observing.get(detector, float("nan")) for detector in detectors],
        dtype=np.float64,
    )

    # The three named losses the coverage decomposition accounts for. Carried separately
    # because they have different causes and different fixes: a window that will not fit
    # inside a segment, a stride phase restarting at each segment, and the boundary holes
    # a chunked release leaves. The search-grade release has none of the last.
    lost_boundary = float(segments.get("lost_boundary_holes_s", 0.0))
    lost_phase = float(segments.get("lost_phase_restart_s", 0.0))
    lost_gaps = max(coincident - analysed - lost_boundary - lost_phase, 0.0)

    longest = float(np.nanmax(observing_s)) if observing_s.size else 0.0
    return FigData(
        figure="livetime_and_duty_cycle",
        arrays={
            "arm": np.array([spec.arm] * len(detectors), dtype=object),
            "observing_s": observing_s,
            "coincident_s": np.full(len(detectors), coincident),
            "analysed_s": np.full(len(detectors), analysed),
            "lost_boundary_s": np.full(len(detectors), lost_boundary),
            "lost_phase_restart_s": np.full(len(detectors), lost_phase),
            "lost_gaps_s": np.full(len(detectors), lost_gaps),
            # Coincident time as a fraction of the longest single-detector observation,
            # which is the reading `pycbc_page_segplot` puts on the same axis.
            "duty_cycle": np.full(
                len(detectors), coincident / longest if longest > 0 else float("nan")
            ),
        },
        scalars={
            "detectors": ",".join(detectors),
            "observing_run": str(spec.data.observing_run),
            "n_windows": int(grid.get("n_windows", 0)),
        },
        attrs={"origin": "pycbc: pycbc_page_segplot"},
    )


def training_prior(spec) -> FigData:
    """
    The parameter distribution the network was trained on.

    Marks the searched region, which bounds where a sensitivity statement applies.
    """
    raise NotImplementedError


def pipeline_diagram(spec) -> FigData:
    """Stage graph and configuration, taken from the stage registry and the spec."""
    raise NotImplementedError


def network_response(spec) -> FigData:
    """The network's output around a known event, showing the trigger's shape."""
    raise NotImplementedError


def calibration(spec) -> FigData:
    """Calibration of the reported probabilities against outcomes."""
    raise NotImplementedError


def build(spec, figures: Optional[list] = None) -> Dict[str, Path]:
    """Build the descriptive figure data products."""
    raise NotImplementedError
