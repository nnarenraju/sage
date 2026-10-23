"""Hierarchical removal of loud zero-lag triggers, as in PyCBC.

While the loudest remaining zero-lag trigger is louder than all of the
background, it is removed along with the background triggers that use
its data.
"""

from typing import Dict, List, Tuple

import numpy as np


def hierarchical_removal(
    background: Dict[str, np.ndarray],
    foreground: Dict[str, np.ndarray],
    detectors: List[str],
    window_s: float = 1.0,
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """Remove loud zero-lag triggers and their data from the background.

    Each step takes the loudest remaining zero-lag trigger. If it is
    louder than every background trigger, the background triggers whose
    merger time in any detector is within ``window_s`` of it are removed,
    and the next loudest is taken. The background time is unchanged.

    Parameters
    ----------
    background : dict of str to numpy.ndarray
        Clustered background triggers, with ``<detector>_gps``.
    foreground : dict of str to numpy.ndarray
        Clustered zero-lag triggers.
    detectors : list of str
        Detectors of the search.
    window_s : float
        Half-width of the removal window around each removed trigger.

    Returns
    -------
    columns : dict of str to numpy.ndarray
        Remaining background triggers.
    removed : list of int
        Indices of the removed zero-lag triggers, in removal order.
    """
    offset = background["tc_gps"] - background["gps"]
    times = [background[f"{detector}_gps"] + offset for detector in detectors]
    keep = np.ones(len(background["stat"]), dtype=bool)

    removed = []
    for index in np.argsort(foreground["stat"])[::-1]:
        loudest_background = background["stat"][keep].max(initial=-np.inf)
        if foreground["stat"][index] <= loudest_background:
            break

        centre = foreground["tc_gps"][index]
        for detector_times in times:
            keep &= np.abs(detector_times - centre) > window_s
        removed.append(int(index))

    return {name: values[keep] for name, values in background.items()}, removed
