"""Cluster triggers by PyCBC's rule: keep the loudest within the window.

The default window is PyCBC's, ``pycbc_coinc_statmap --cluster-window``.
"""

from typing import Dict, List

import numpy as np
from pycbc.events.coinc import cluster_over_time

from sage.search.zerolag.triggers import merger_times


def cluster_triggers(
    columns: Dict[str, np.ndarray], window_s: float = 10.0
) -> Dict[str, np.ndarray]:
    """Zero-lag or single-slide triggers, one per event.

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
    keep = cluster_over_time(columns["stat"], columns["tc_gps"], window_s)
    return {name: values[keep] for name, values in columns.items()}


def cluster_pairs(
    columns: Dict[str, np.ndarray], detectors: List[str], window_s: float = 10.0
) -> Dict[str, np.ndarray]:
    """Random-pair triggers, one per event.

    A pair is kept if it is the loudest of the pairs whose merger times
    are within the window of its own in every detector.

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
    stats = columns["stat"][order]
    left = np.searchsorted(times[:, 0], times[:, 0] - window_s)
    right = np.searchsorted(times[:, 0], times[:, 0] + window_s)

    keep = []
    for index in range(len(order)):
        rivals = np.arange(left[index], right[index])
        difference = times[rivals] - times[index]
        close = np.all((difference >= -window_s) & (difference < window_s), axis=1)
        rivals = rivals[close]
        if rivals[np.argmax(stats[rivals])] == index:
            keep.append(index)

    keep = order[np.array(keep, dtype=np.int64)]
    return {name: values[keep] for name, values in columns.items()}
