"""Flag windows whose data touches bad time, as PyCBC vetoes them.

Bad time fails CBC Category 2 or holds hardware injections; flagged
triggers stay in the search, and are left out of significance and live time.
"""

from typing import Dict, List

import numpy as np

from sage.data.noise.data_quality import Interval, overlaps


def flag_vetoed(
    network: Dict[str, object],
    columns: Dict[str, np.ndarray],
    vetoes: Dict[str, List[Interval]],
) -> np.ndarray:
    """Windows whose analysed data in any detector overlaps its bad time.

    A detector's analysed data starts one padding after its window start
    and stops one padding before its end; any overlap flags the window.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    columns : dict of str to numpy.ndarray
        Window or trigger columns with ``gps`` and ``<detector>_gps``.
    vetoes : dict of str to list of tuple of float
        ``sage.data.noise.data_quality.bad_segments`` per detector.

    Returns
    -------
    numpy.ndarray
        True for each flagged window.
    """
    padding_s = network["padding_s"]
    content_s = network["window_samples"] / network["sample_rate"] - 2 * padding_s

    flagged = np.zeros(len(columns["gps"]), dtype=bool)
    for detector in network["detectors"]:
        start = columns[f"{detector}_gps"] + padding_s
        flagged |= overlaps(start, start + content_s, vetoes[detector])
    return flagged
