"""Cluster triggers by greedy non-maximum suppression.

The loudest trigger is kept and its neighbours removed, repeatedly
(Girshick et al. 2014; Hosang et al. 2017). Windows and ties follow PyCBC.
"""

from typing import Dict, List

import numpy as np

from sage.search.zerolag.triggers import merger_times


def greedy_triggers(
    columns: Dict[str, np.ndarray], window_s: float = 10.0
) -> Dict[str, np.ndarray]:
    """Zero-lag or single-slide triggers, one per event.

    The loudest trigger is kept and every trigger within the window of its
    merger time removed, until none is left. The window is ``[t - w, t + w)``
    and ties go to the earlier trigger, as in PyCBC.

    Parameters
    ----------
    columns : dict of str to numpy.ndarray
        Trigger columns of one zero-lag run or one time slide.
    window_s : float
        Clustering window around each merger time, in seconds.

    Returns
    -------
    dict of str to numpy.ndarray
        Columns of the kept triggers, in time order.
    """
    by_time = np.argsort(columns["tc_gps"], kind="stable")
    times = columns["tc_gps"][by_time]
    left = np.searchsorted(times, times - window_s)
    right = np.searchsorted(times, times + window_s)

    removed = np.zeros(len(times), dtype=bool)
    keep = []
    for index in np.lexsort((times, -columns["stat"][by_time])):
        if removed[index]:
            continue
        keep.append(index)
        removed[left[index] : right[index]] = True

    keep = by_time[np.sort(np.array(keep, dtype=np.int64))]
    return {name: values[keep] for name, values in columns.items()}


def greedy_pairs(
    columns: Dict[str, np.ndarray], detectors: List[str], window_s: float = 10.0
) -> Dict[str, np.ndarray]:
    """Random-pair triggers, one per event.

    The loudest pair is kept and every pair within the window of its merger
    times in every detector removed, until none is left.

    Parameters
    ----------
    columns : dict of str to numpy.ndarray
        Trigger columns of random pairs, with ``<detector>_gps``.
    detectors : list of str
        Detectors of the pairs.
    window_s : float
        Clustering window around each merger time, in seconds.

    Returns
    -------
    dict of str to numpy.ndarray
        Columns of the kept pairs, in reference time order.
    """
    times = merger_times(columns, detectors)
    order = np.argsort(times[:, 0], kind="stable")
    times = times[order]
    left = np.searchsorted(times[:, 0], times[:, 0] - window_s)
    right = np.searchsorted(times[:, 0], times[:, 0] + window_s)

    removed = np.zeros(len(order), dtype=bool)
    keep = []
    for index in np.lexsort((times[:, 0], -columns["stat"][order])):
        if removed[index]:
            continue
        keep.append(index)
        rivals = np.arange(left[index], right[index])
        difference = times[rivals] - times[index]
        close = np.all((difference >= -window_s) & (difference < window_s), axis=1)
        removed[rivals[close]] = True

    keep = order[np.sort(np.array(keep, dtype=np.int64))]
    return {name: values[keep] for name, values in columns.items()}
