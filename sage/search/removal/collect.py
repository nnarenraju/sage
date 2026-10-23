"""Read trigger files as clustered triggers and the time they cover.

Vetoed triggers and their time are left out. Each file is clustered on its
own, since each time slide is a separate background run; background times
are summed over files.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from sage.search.clustering.greedy import greedy_pairs, greedy_triggers
from sage.search.clustering.sliding_window import cluster_pairs, cluster_triggers
from sage.search.zerolag.triggers import concatenate, read_triggers


def load_clustered(
    path: Path, greedy: bool, window_s: float = 10.0
) -> Tuple[Dict[str, np.ndarray], float]:
    """Clustered triggers of one file clear of bad time, and that time.

    Vetoed triggers are dropped before clustering, as PyCBC drops them
    before forming coincidences.

    Parameters
    ----------
    path : Path
        Zero-lag, time-slide or random-pair trigger file.
    greedy : bool
        Cluster by greedy suppression rather than PyCBC's rule.
    window_s : float
        Clustering window in seconds; PyCBC's default.

    Returns
    -------
    columns : dict of str to numpy.ndarray
        Clustered trigger columns.
    time_s : float
        Analysed or background time of the file clear of bad time, in
        seconds.
    """
    columns, attrs = read_triggers(path)
    kept = ~columns["vetoed"]
    columns = {name: values[kept] for name, values in columns.items()}
    if attrs.get("method") == "random_pairs":
        cluster = greedy_pairs if greedy else cluster_pairs
        columns = cluster(columns, list(attrs["detectors"]), window_s)
    else:
        cluster = greedy_triggers if greedy else cluster_triggers
        columns = cluster(columns, window_s)

    time_s = attrs["n_live_windows"] * attrs["stride_samples"] / attrs["sample_rate"]
    return columns, float(time_s)


def load_background(
    paths: List[Path], greedy: bool, window_s: float = 10.0
) -> Tuple[Dict[str, np.ndarray], float]:
    """Clustered triggers of every background file, joined.

    Parameters
    ----------
    paths : list of Path
        Time-slide or random-pair trigger files.
    greedy : bool
        Cluster by greedy suppression rather than PyCBC's rule.
    window_s : float
        Clustering window in seconds; PyCBC's default.

    Returns
    -------
    columns : dict of str to numpy.ndarray
        Clustered background triggers of all files.
    time_s : float
        Total background time in seconds.
    """
    parts = []
    total_s = 0.0
    for path in paths:
        columns, time_s = load_clustered(path, greedy, window_s)
        parts.append(columns)
        total_s += time_s
    return concatenate(parts), total_s
