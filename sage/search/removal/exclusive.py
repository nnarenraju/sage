"""Exclusive background, as in PyCBC.

Background triggers that use data within a short window of any zero-lag
trigger, in any detector, are removed.
"""

from typing import Dict, List, Tuple

import numpy as np


def near(times: np.ndarray, centres: np.ndarray, window_s: float) -> np.ndarray:
    """Whether each time is within ``window_s`` of any centre.

    Parameters
    ----------
    times : numpy.ndarray
        Times to test.
    centres : numpy.ndarray
        Centre times, ascending.
    window_s : float
        Half-width of the window around each centre.

    Returns
    -------
    numpy.ndarray
        Boolean mask over ``times``.
    """
    after = np.clip(np.searchsorted(centres, times), 1, len(centres) - 1)
    before = after - 1
    distance = np.minimum(
        np.abs(times - centres[before]), np.abs(times - centres[after])
    )
    return distance <= window_s


def exclusive_background(
    background: Dict[str, np.ndarray],
    background_time_s: float,
    foreground: Dict[str, np.ndarray],
    foreground_time_s: float,
    detectors: List[str],
    window_s: float = 0.1,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Background without triggers near any zero-lag trigger.

    A background trigger is removed if its merger time in any detector
    is within ``window_s`` of a zero-lag merger time. As in PyCBC, the
    vetoed time is removed from every slide, so the background time
    shrinks by the vetoed fraction of the zero-lag time.

    Parameters
    ----------
    background : dict of str to numpy.ndarray
        Clustered background triggers, with ``<detector>_gps``.
    background_time_s : float
        Background time in seconds.
    foreground : dict of str to numpy.ndarray
        Clustered zero-lag triggers.
    foreground_time_s : float
        Zero-lag analysed time in seconds.
    detectors : list of str
        Detectors of the search.
    window_s : float
        Half-width of the veto window around each zero-lag trigger.

    Returns
    -------
    columns : dict of str to numpy.ndarray
        Remaining background triggers.
    time_s : float
        Remaining background time in seconds.
    """
    centres = np.sort(foreground["tc_gps"])
    offset = background["tc_gps"] - background["gps"]

    keep = np.ones(len(background["stat"]), dtype=bool)
    for detector in detectors:
        keep &= ~near(background[f"{detector}_gps"] + offset, centres, window_s)

    # Length of the union of the veto windows.
    veto_s = 2 * window_s + np.minimum(np.diff(centres), 2 * window_s).sum()
    time_s = background_time_s * (1 - veto_s / foreground_time_s)
    return {name: values[keep] for name, values in background.items()}, time_s
