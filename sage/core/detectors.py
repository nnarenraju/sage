#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Light-travel times between detectors, from :mod:`pycbc.detector`."""

import numpy as np


def light_travel_time(det_a: str, det_b: str) -> float:
    """
    Light-travel time between two detectors, in seconds.

    The largest possible difference in signal arrival time between the
    two sites.

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
    # pycbc is optional (the lal extra); imported only when needed
    from pycbc.detector import Detector

    return float(Detector(det_a).light_travel_time_to_detector(Detector(det_b)))


def pairwise_light_travel_times(detectors) -> np.ndarray:
    """
    Symmetric matrix of pairwise light-travel times, in seconds.

    Entry ``[i, j]`` is the time between ``detectors[i]`` and
    ``detectors[j]``; the diagonal is zero.

    Parameters
    ----------
    detectors : sequence of str
        Detector names in network order.

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
