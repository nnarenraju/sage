"""Time-slide background by rolling the window list, as in sgwc-1.

Roll ``j`` pairs each reference window ``i`` with follower ``f``'s window
``(i + j (f + 1) k) mod N`` of the zero-lag list, ``k`` windows spanning
at least the step between rolls. Rolls are grouped into files of about
a year of live time; a file's rolls share each window's frontend.
"""

import math
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from pycbc.conversions import sec_to_year

from sage.core.progress import bar
from sage.search.dataprep.strain import batch_bounds, window_batches
from sage.search.model.score import (
    backend_outputs,
    check_separable,
    frontend_features,
    window_spectra,
)
from sage.search.records import record_done, start_stream, stream_files, write_rows


def roll_shifts(
    n_rolls: int,
    roll_windows: int,
    n_followers: int,
    min_separation_s: float = 100.0,
    stride_samples: int = 204,
    sample_rate: float = 2048.0,
) -> np.ndarray:
    """Window shift of each other detector for each roll.

    Roll ``j`` shifts detector ``f`` by ``j (f + 1) roll_windows`` windows,
    so every detector pair is at least one roll length apart.

    Parameters
    ----------
    n_rolls : int
        Number of rolls.
    roll_windows : int
        Roll length in windows.
    n_followers : int
        Number of detectors other than the reference.
    min_separation_s : float
        Smallest time between paired windows.
    stride_samples : int
        Step between window starts in samples.
    sample_rate : float
        Sample rate in Hz.

    Returns
    -------
    numpy.ndarray
        Shifts in windows, shape ``(n_rolls, n_followers)``.

    Raises
    ------
    ValueError
        If one roll is shorter than ``min_separation_s``.
    """
    roll_s = roll_windows * stride_samples / sample_rate
    if roll_s < min_separation_s:
        raise ValueError(f"roll of {roll_s} s is under {min_separation_s} s")

    rolls = np.arange(1, n_rolls + 1)[:, None]
    return roll_windows * rolls * np.arange(1, n_followers + 1)


def gather(
    chunks: List[Tuple[int, torch.Tensor]], start: int, count: int
) -> torch.Tensor:
    """Features of ``count`` consecutive windows from stored chunks.

    Parameters
    ----------
    chunks : list of tuple of int and torch.Tensor
        First window index and features of each stored chunk, in order.
    start : int
        Index of the first window wanted.
    count : int
        Number of windows wanted.

    Returns
    -------
    torch.Tensor
        Features of the wanted windows.
    """
    pieces = []
    for first, features in chunks:
        begin = max(start - first, 0)
        end = min(start + count - first, len(features))
        if begin < end:
            pieces.append(features[begin:end])
    return torch.cat(pieces)


def step_windows(step_s: float, stride_samples: int, sample_rate: float) -> int:
    """Fewest whole windows whose starts span at least ``step_s``.

    Parameters
    ----------
    step_s : float
        Smallest time between rolls, in seconds.
    stride_samples : int
        Step between window starts in samples.
    sample_rate : float
        Sample rate in Hz.

    Returns
    -------
    int
        Windows per roll step.
    """
    return math.ceil(Fraction(step_s) * Fraction(sample_rate) / stride_samples)


def plan_rolls(
    clear: Dict[str, np.ndarray],
    detectors: List[str],
    step: int,
    step_s: float,
    live_time_years: float,
    file_years: float,
    stride_samples: int,
    sample_rate: float,
) -> List[dict]:
    """Rolls 1, 2, ... grouped into files, until the live time is reached.

    A roll's live time is its pairs clear of bad time in every detector
    times the stride; a file is closed once its live time reaches
    ``file_years``.

    Parameters
    ----------
    clear : dict of str to numpy.ndarray
        ``clear_windows`` of the zero-lag windows.
    detectors : list of str
        Detectors, the reference first.
    step : int
        Windows per roll step, ``step_windows``.
    step_s : float
        Smallest time between rolls, in seconds.
    live_time_years : float
        Live time of the whole background, in years.
    file_years : float
        Live time after which a file is closed, in years.
    stride_samples : int
        Step between window starts in samples.
    sample_rate : float
        Sample rate in Hz.

    Returns
    -------
    list of dict
        Per file, ``shifts`` (one row per roll, one shift per follower)
        and ``live_time_s`` (per roll).

    Raises
    ------
    ValueError
        If a roll has no pair clear of bad time.
    """
    reference, followers = detectors[0], detectors[1:]
    stride_s = stride_samples / sample_rate
    files, total_s, roll = [], 0.0, 0
    while sec_to_year(total_s) < live_time_years:
        roll += 1
        shifts = roll_shifts(
            roll, step, len(followers), step_s, stride_samples, sample_rate
        )[-1]
        pairs = clear[reference].copy()
        for follower, shift in zip(followers, shifts):
            pairs &= np.roll(clear[follower], -shift)
        if not pairs.any():
            raise ValueError(f"roll {roll} has no pair clear of bad time")

        live_s = int(pairs.sum()) * stride_s
        if not files or sec_to_year(sum(files[-1]["live_time_s"])) >= file_years:
            files.append({"shifts": [], "live_time_s": []})
        files[-1]["shifts"].append([int(shift) for shift in shifts])
        files[-1]["live_time_s"].append(live_s)
        total_s += live_s
    return files


def rotated_windows(
    windows: Dict[str, np.ndarray], offset: int, extra: int
) -> Dict[str, np.ndarray]:
    """The window list from window ``offset`` on, wrapping, ``extra`` longer.

    Parameters
    ----------
    windows : dict of str to numpy.ndarray
        Output of ``analysis_windows``.
    offset : int
        Window at position 0.
    extra : int
        Windows after the full list, wrapping again.

    Returns
    -------
    dict of str to numpy.ndarray
        Windows at positions ``0 .. N + extra``.
    """
    n_windows = len(windows["gps"])
    positions = (offset + np.arange(n_windows + extra)) % n_windows
    return {name: values[positions] for name, values in windows.items()}


def score_roll_file(
    network: Dict[str, object],
    release_dir: Path,
    run: str,
    windows: Dict[str, np.ndarray],
    directory: Path,
    name: str,
    shifts: List[List[int]],
    stride_samples: int,
    batch_size: int,
    device: str,
    attrs: dict,
) -> None:
    """Score the rolls of one background file, resuming after the last batch.

    Row ``r * N + i`` holds roll ``r`` of the file: reference window ``i``
    with follower ``f``'s window ``(i + shifts[r][f]) mod N``. Each
    window's frontend features are computed once for the file.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    release_dir : Path
        Strain release directory.
    run : str
        Observing run as the release names it, e.g. ``O3a``.
    windows : dict of str to numpy.ndarray
        Zero-lag windows, from ``analysis_windows``.
    directory : Path
        Background directory of the search.
    name : str
        File name.
    shifts : list of list of int
        Per roll, the shift of each follower in windows.
    stride_samples : int
        Step between window starts in samples.
    batch_size : int
        Maximum windows scored at once.
    device : str
        Device to score on.
    attrs : dict
        Settings recorded with the scores; a resumed pass must match them.
    """
    detectors = network["detectors"]
    window_samples = network["window_samples"]
    n_windows = len(windows["gps"])
    rows = len(shifts) * n_windows
    done = start_stream(directory, name, network["columns"], rows, attrs)
    if done == n_windows:
        return

    # Each follower's windows from its smallest shift on, read in batches
    # that do not depend on where a job stopped.
    shifts = np.asarray(shifts)
    offsets = shifts.min(axis=0)
    spans = shifts.max(axis=0) - offsets
    followers = []
    for offset, span in zip(offsets, spans):
        rotated = rotated_windows(windows, offset, span)
        bounds = batch_bounds(rotated, detectors, stride_samples, batch_size)
        start = max(first for first, _ in bounds if first <= done)
        batches = window_batches(
            release_dir,
            run,
            rotated,
            detectors,
            window_samples,
            stride_samples,
            batch_size,
            start=start,
        )
        followers.append({"batches": batches, "chunks": [], "end": start})

    checked = False
    progress = bar(total=n_windows, initial=done, unit="window")
    with progress, open(stream_files(directory, name)[0], "r+b") as handle:
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
            if not checked:
                check_separable(network, spectra)
                progress.write("  split network checked: exact, detectors separate")
                checked = True
            reference = frontend_features(network, spectra, [0])

            for channel, (follower, span) in enumerate(zip(followers, spans), 1):
                while follower["end"] < batch.stop + span:
                    position, follower_strain = next(follower["batches"])
                    follower_spectra = window_spectra(
                        follower_strain, window_samples, stride_samples, device
                    )
                    features = frontend_features(network, follower_spectra, [channel])
                    follower["chunks"].append((position.start, features[0]))
                    follower["end"] = position.stop

            count = batch.stop - batch.start
            for row, shift in enumerate(shifts):
                paired = reference + [
                    gather(follower["chunks"], batch.start + ahead - offset, count)
                    for follower, ahead, offset in zip(followers, shift, offsets)
                ]
                first_row = row * n_windows + batch.start
                write_rows(handle, first_row, backend_outputs(network, paired))
            record_done(handle, directory, name, batch.stop)

            for follower in followers:
                follower["chunks"] = [
                    (first, features)
                    for first, features in follower["chunks"]
                    if first + len(features) > batch.stop
                ]
            progress.update(batch.stop - batch.start)
