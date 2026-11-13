"""Score every zero-lag window: all network outputs, resumable.

``score_windows`` keeps only loud windows as triggers, for the time-lag
background.
"""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from sage.core.progress import bar
from sage.search.dataprep.strain import window_batches
from sage.search.model.score import network_outputs, score_strain, window_spectra
from sage.search.records import record_done, start_stream, stream_files, write_rows
from sage.search.zerolag.triggers import concatenate, loud_triggers

WINDOW_TABLE = "windows.npy"
SCORES = "scores"


def score_windows(
    network: Dict[str, object],
    release_dir: Path,
    run: str,
    windows: Dict[str, np.ndarray],
    stride_samples: int = 204,
    keep_threshold: float = 0.0,
    batch_size: int = 4096,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Score windows and return the loud ones as trigger columns.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    release_dir : Path
        Strain release directory.
    run : str
        Observing run.
    windows : dict of str to numpy.ndarray
        Output of ``analysis_windows``.
    stride_samples : int
        Step between window starts in samples.
    keep_threshold : float
        Ranking statistic above which a window is kept as a trigger.
    batch_size : int
        Maximum windows scored at once.
    device : str
        Device to score on.

    Returns
    -------
    dict of str to numpy.ndarray
        Trigger columns of the kept windows.
    """
    kept = []
    for batch, strain in window_batches(
        release_dir,
        run,
        windows,
        network["detectors"],
        network["window_samples"],
        stride_samples,
        batch_size,
    ):
        statistic, estimates = score_strain(
            network, strain, stride_samples, device
        )
        scored = {name: values[batch] for name, values in windows.items()}
        kept.append(
            loud_triggers(network, statistic, estimates, scored, keep_threshold)
        )
    return concatenate(kept)


def window_table(windows: Dict[str, np.ndarray], detectors: List[str]) -> np.ndarray:
    """
    Table of each window's GPS start, segments and start samples.

    Parameters
    ----------
    windows : dict of str to numpy.ndarray
        Output of ``analysis_windows``.
    detectors : list of str
        Detectors of the network.

    Returns
    -------
    numpy.ndarray
        Structured array, one row per window.
    """
    fields = [("gps", "<f8")]
    fields += [(f"{detector}_segment", "<i8") for detector in detectors]
    fields += [(f"{detector}_sample", "<i8") for detector in detectors]
    table = np.empty(len(windows["gps"]), dtype=fields)
    for name, _ in fields:
        table[name] = windows[name]
    return table


def save_window_table(path: Path, table: np.ndarray) -> None:
    """
    Save the window table once; later calls check it is unchanged.

    Parameters
    ----------
    path : Path
        Table file, ``.npy``.
    table : numpy.ndarray
        Output of ``window_table``.

    Raises
    ------
    ValueError
        If the saved table differs, so the release has changed.
    """
    if path.exists():
        if not np.array_equal(np.load(path), table):
            raise ValueError(f"{path}: the strain release's windows have changed")
        return

    part = path.with_name(f"{path.name}.part")
    with open(part, "wb") as handle:
        np.save(handle, table)
    os.replace(part, path)


def score_zerolag(
    network: Dict[str, object],
    release_dir: Path,
    run: str,
    windows: Dict[str, np.ndarray],
    directory: Path,
    stride_samples: int,
    batch_size: int,
    device: str,
    attrs: dict,
) -> None:
    """
    Score every zero-lag window, resuming after the last batch written.

    Writes ``windows.npy`` and the ``scores`` stream (every network
    output, row ``i`` for window ``i``) into ``directory``.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    release_dir : Path
        Strain release directory.
    run : str
        Observing run as the release names it, e.g. ``O3a``.
    windows : dict of str to numpy.ndarray
        Output of ``analysis_windows``.
    directory : Path
        Zero-lag directory of the search.
    stride_samples : int
        Step between window starts in samples.
    batch_size : int
        Maximum windows scored at once.
    device : str
        Device to score on.
    attrs : dict
        Settings recorded with the scores; a resumed pass must match
        them.

    Raises
    ------
    SettingsChanged
        If the scores were started with other columns, rows or
        settings.
    ValueError
        If the saved window table differs from ``windows``.
    """
    detectors = network["detectors"]
    window_samples = network["window_samples"]
    total = len(windows["gps"])
    directory.mkdir(parents=True, exist_ok=True)
    save_window_table(directory / WINDOW_TABLE, window_table(windows, detectors))

    done = start_stream(directory, SCORES, network["columns"], total, attrs)
    progress = bar(total=total, initial=done, unit="window")
    with progress, open(stream_files(directory, SCORES)[0], "r+b") as handle:
        for batch, strain in window_batches(
            release_dir,
            run,
            windows,
            detectors,
            window_samples,
            stride_samples,
            batch_size,
            start=done,
        ):
            spectra = window_spectra(strain, window_samples, stride_samples, device)
            write_rows(handle, batch.start, network_outputs(network, spectra))
            record_done(handle, directory, SCORES, batch.stop)
            progress.update(batch.stop - batch.start)
