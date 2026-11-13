"""Flag windows whose analysed data touches bad time.

Bad time fails CBC Category 2, holds hardware injections or exactly-zero
strain. A detector's analysed data is the window without its padding at
either end; any overlap with bad time flags the window (user ruling
2026-11-13). Flagged windows stay scored, and are left out of
significance and live time.
"""

from typing import Dict, List

import numpy as np

from sage.data.noise.data_quality import Interval, overlaps


def clear_windows(
    network: Dict[str, object],
    columns: Dict[str, np.ndarray],
    vetoes: Dict[str, List[Interval]],
) -> Dict[str, np.ndarray]:
    """Per detector, whether each window's analysed data is clear of bad time.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    columns : dict of str to numpy.ndarray
        Window or trigger columns with ``<detector>_gps``.
    vetoes : dict of str to list of tuple of float
        Bad time per detector, ``read_bad_time`` of its strain.

    Returns
    -------
    dict of str to numpy.ndarray
        True for each clear window, per detector.
    """
    padding_s = network["padding_s"]
    content_s = network["window_samples"] / network["sample_rate"] - 2 * padding_s

    clear = {}
    for detector in network["detectors"]:
        start = columns[f"{detector}_gps"] + padding_s
        clear[detector] = ~overlaps(start, start + content_s, vetoes[detector])
    return clear


def flag_vetoed(
    network: Dict[str, object],
    columns: Dict[str, np.ndarray],
    vetoes: Dict[str, List[Interval]],
) -> np.ndarray:
    """Windows whose analysed data in any detector overlaps its bad time.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    columns : dict of str to numpy.ndarray
        Window or trigger columns with ``<detector>_gps``.
    vetoes : dict of str to list of tuple of float
        Bad time per detector, ``read_bad_time`` of its strain.

    Returns
    -------
    numpy.ndarray
        True for each flagged window.
    """
    clear = clear_windows(network, columns, vetoes).values()
    return ~np.logical_and.reduce(list(clear))
