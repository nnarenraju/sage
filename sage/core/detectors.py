#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Detector-geometry helpers.

Thin wrappers over :mod:`pycbc.detector` for quantities that depend only on the
detector network (independent of source/sky position), e.g. the light-travel
time between a pair of detectors. Used by the multi-detector consistency head to
bound the physically-allowed inter-detector arrival-time difference.
"""

import numpy as np


def light_travel_time(det_a: str, det_b: str) -> float:
    """Exact light-travel time (seconds) between two detectors.

    This is the maximum possible difference in signal arrival time between the
    two sites (a source on the line joining them), taken from pycbc's LAL-backed
    detector geometry.

    Parameters
    ----------
    det_a, det_b : str
        Detector names, e.g. ``"H1"``, ``"L1"``, ``"V1"``.

    Returns
    -------
    float
        Light-travel time in seconds (``0.0`` if ``det_a == det_b``).
    """
    if det_a == det_b:
        return 0.0
    # pycbc is an optional dependency (declared under the `lal` extra); import it
    # lazily so importing this module / the consistency network does not require
    # it — only computing a light-travel time does.
    from pycbc.detector import Detector

    return float(Detector(det_a).light_travel_time_to_detector(Detector(det_b)))


def pairwise_light_travel_times(detectors) -> np.ndarray:
    """Symmetric ``(D, D)`` matrix of pairwise light-travel times (seconds).

    Entry ``[i, j]`` is the light-travel time between ``detectors[i]`` and
    ``detectors[j]``; the diagonal is zero. For the common two-detector network
    the single relevant value is ``[0, 1]``.

    Parameters
    ----------
    detectors : sequence of str
        Detector names in network order (typically ``cfg.detectors``).

    Returns
    -------
    np.ndarray, shape ``(D, D)``, float64
    """
    n = len(detectors)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            ltt = light_travel_time(detectors[i], detectors[j])
            out[i, j] = ltt
            out[j, i] = ltt
    return out
