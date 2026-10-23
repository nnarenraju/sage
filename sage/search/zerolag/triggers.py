"""Trigger columns from the network's point estimates, and trigger files.

One HDF5 file per GPU job: one dataset per trigger column, and the
job's settings as file attributes.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


def decode(
    estimates: np.ndarray,
    offsets: np.ndarray,
    scales: np.ndarray,
    sigma_min: float = 1e-3,
    sigma_max: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Physical means and widths from the point-estimate head.

    The width follows the training loss: softplus of the raw output plus
    ``sigma_min``, clipped to ``[sigma_min, sigma_max]``.

    Parameters
    ----------
    estimates : numpy.ndarray
        Head output, shape ``(windows, 2 * targets)``: means, then raw
        widths.
    offsets, scales : numpy.ndarray
        Encoding constants from ``encoding``.
    sigma_min, sigma_max : float
        Width bounds used by the training loss.

    Returns
    -------
    means, widths : numpy.ndarray
        Physical values and standard deviations, shape
        ``(windows, targets)``.

    Raises
    ------
    ValueError
        If the head output contains non-finite values.
    """
    estimates = np.asarray(estimates, dtype=np.float64)
    if not np.isfinite(estimates).all():
        raise ValueError("the point-estimate head returned non-finite values")

    n_targets = len(offsets)
    means = estimates[:, :n_targets] * scales + offsets
    raw_widths = estimates[:, n_targets:]
    widths = np.clip(np.logaddexp(0.0, raw_widths) + sigma_min, sigma_min, sigma_max)
    return means, widths * scales


def trigger_columns(
    estimates: np.ndarray,
    window_start_gps: np.ndarray,
    targets: List[str],
    offsets: np.ndarray,
    scales: np.ndarray,
    padding_s: float,
) -> Dict[str, np.ndarray]:
    """Point-estimate columns of a trigger file.

    The merger time is made absolute: ``tc_gps = window_start + padding
    + tc``, since tc is measured from the end of the leading padding.

    Parameters
    ----------
    estimates : numpy.ndarray
        Head output, shape ``(windows, 2 * targets)``.
    window_start_gps : numpy.ndarray
        Float64 GPS start of each window.
    targets : list of str
        Point-estimate targets in head order.
    offsets, scales : numpy.ndarray
        Encoding constants from ``encoding``.
    padding_s : float
        Whitening padding at each end of a window, in seconds.

    Returns
    -------
    dict of str to numpy.ndarray
        ``<target>`` and ``<target>_sigma`` per target, with ``tc``
        replaced by ``tc_gps``.
    """
    means, widths = decode(estimates, offsets, scales)

    columns = {}
    for index, name in enumerate(targets):
        columns[name] = means[:, index]
        columns[f"{name}_sigma"] = widths[:, index]

    columns["tc_gps"] = window_start_gps + (padding_s + columns.pop("tc"))
    return columns


def loud_triggers(
    network: Dict[str, object],
    statistic: np.ndarray,
    estimates: np.ndarray,
    windows: Dict[str, np.ndarray],
    keep_threshold: float,
) -> Dict[str, np.ndarray]:
    """Trigger columns of the windows louder than a threshold.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    statistic, estimates : numpy.ndarray
        Network output for the scored windows.
    windows : dict of str to numpy.ndarray
        ``gps`` and each detector's segment and sample for the scored
        windows, in the format of ``analysis_windows``.
    keep_threshold : float
        Ranking statistic above which a window is kept.

    Returns
    -------
    dict of str to numpy.ndarray
        Decoded point estimates, ``stat`` and the window columns.
    """
    loud = statistic > keep_threshold
    columns = trigger_columns(
        estimates[loud],
        windows["gps"][loud],
        network["targets"],
        network["offsets"],
        network["scales"],
        network["padding_s"],
    )
    columns["stat"] = statistic[loud]
    for name, values in windows.items():
        columns[name] = values[loud]
    return columns


def merger_times(
    columns: Dict[str, np.ndarray], detectors: List[str]
) -> np.ndarray:
    """Merger time of each trigger in every detector.

    A detector's merger time is its window start plus the merger offset in
    the window, as in sgwc-1.

    Parameters
    ----------
    columns : dict of str to numpy.ndarray
        Trigger columns with ``gps``, ``tc_gps`` and ``<detector>_gps``.
    detectors : list of str
        Detectors of the triggers.

    Returns
    -------
    numpy.ndarray
        Merger times, shape ``(triggers, detectors)``.
    """
    offset = columns["tc_gps"] - columns["gps"]
    return np.stack([columns[f"{detector}_gps"] + offset for detector in detectors], 1)


def concatenate(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Join trigger columns from consecutive batches.

    Parameters
    ----------
    parts : list of dict of str to numpy.ndarray
        Trigger columns of each batch, with the same names.

    Returns
    -------
    dict of str to numpy.ndarray
        Columns joined in order.
    """
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def write_triggers(
    path: Path, columns: Dict[str, np.ndarray], attrs: Dict[str, object]
) -> None:
    """Write trigger columns and attributes to one HDF5 file.

    Parameters
    ----------
    path : Path
        Output file; replaced if it exists.
    columns : dict of str to numpy.ndarray
        Trigger columns of equal length.
    attrs : dict of str to object
        Settings stored as file attributes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for name, values in columns.items():
            handle.create_dataset(name, data=np.asarray(values))
        handle.attrs.update(attrs)


def read_triggers(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Read trigger columns and attributes from one HDF5 file.

    Parameters
    ----------
    path : Path
        Trigger file.

    Returns
    -------
    columns : dict of str to numpy.ndarray
        Trigger columns.
    attrs : dict of str to object
        File attributes.
    """
    with h5py.File(path, "r") as handle:
        columns = {name: handle[name][:] for name in handle}
        attrs = dict(handle.attrs)
    return columns, attrs
