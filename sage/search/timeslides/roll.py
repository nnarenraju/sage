"""Time-slide background by rolling the window list, as in sgwc-1.

Each other detector's windows are shifted along the zero-lag window
list by a whole number of windows, wrapping at the end.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from sage.data.noise.data_quality import Interval
from sage.search.dataprep.strain import window_batches
from sage.search.dataprep.windows import analysis_windows, load_segments
from sage.search.model.score import backend_scores, frontend_features, window_spectra
from sage.search.removal.vetoes import flag_vetoed
from sage.search.zerolag.triggers import concatenate, loud_triggers, write_triggers


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


def score_slide(
    network: Dict[str, object],
    windows: Dict[str, np.ndarray],
    stored: List[List[Tuple[int, torch.Tensor]]],
    reference: slice,
    shifts: np.ndarray,
    keep_threshold: float,
) -> Dict[str, np.ndarray]:
    """Trigger columns of one batch of reference windows under one roll.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    windows : dict of str to numpy.ndarray
        Zero-lag windows, from ``analysis_windows``.
    stored : list of list of tuple of int and torch.Tensor
        Feature chunks per detector.
    reference : slice
        Reference windows in the batch.
    shifts : numpy.ndarray
        Shift of every detector in windows, zero for the reference.
    keep_threshold : float
        Ranking statistic above which a window is kept.

    Returns
    -------
    dict of str to numpy.ndarray
        Trigger columns of the kept windows.
    """
    count = reference.stop - reference.start
    features = [
        gather(chunks, reference.start + shift, count)
        for chunks, shift in zip(stored, shifts)
    ]
    statistic, estimates = backend_scores(network, features)

    ordinals = np.arange(reference.start, reference.stop)
    paired = {"gps": windows["gps"][reference]}
    for detector, shift in zip(network["detectors"], shifts):
        partner = (ordinals + shift) % len(windows["gps"])
        paired[f"{detector}_segment"] = windows[f"{detector}_segment"][partner]
        paired[f"{detector}_sample"] = windows[f"{detector}_sample"][partner]
        paired[f"{detector}_gps"] = windows[f"{detector}_gps"][partner]
    return loud_triggers(network, statistic, estimates, paired, keep_threshold)


def score_rolls(
    network: Dict[str, object],
    release_dir: Path,
    run: str,
    windows: Dict[str, np.ndarray],
    shifts: np.ndarray,
    stride_samples: int = 204,
    keep_threshold: float = 0.0,
    batch_size: int = 4096,
    device: str = "cuda",
) -> List[Dict[str, np.ndarray]]:
    """Score every roll with each window's frontend computed once.

    Features are held for the largest shift ahead of the windows being
    scored. The first windows are read again after the last, so shifts
    that wrap find their features.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    release_dir : Path
        Strain release directory.
    run : str
        Observing run.
    windows : dict of str to numpy.ndarray
        Zero-lag windows, from ``analysis_windows``.
    shifts : numpy.ndarray
        Output of ``roll_shifts``.
    stride_samples : int
        Step between window starts in samples.
    keep_threshold : float
        Ranking statistic above which a window is kept.
    batch_size : int
        Maximum windows scored at once.
    device : str
        Device to score on.

    Returns
    -------
    list of dict of str to numpy.ndarray
        Trigger columns of each slide.
    """
    detectors = network["detectors"]
    n_windows = len(windows["gps"])
    max_shift = int(shifts.max())
    extended = {
        name: np.concatenate([values, values[:max_shift]])
        for name, values in windows.items()
    }

    stored = [[] for _ in detectors]
    pending = []
    kept = [[] for _ in shifts]
    for batch, strain in window_batches(
        release_dir,
        run,
        extended,
        detectors,
        network["window_samples"],
        stride_samples,
        batch_size,
    ):
        spectra = window_spectra(
            strain, network["window_samples"], stride_samples, device
        )
        detector_features = frontend_features(network, spectra)
        for chunks, features in zip(stored, detector_features):
            chunks.append((batch.start, features))
        if batch.start < n_windows:
            pending.append(batch)

        # A batch is scored once its partners' features exist.
        while pending and pending[0].stop + max_shift <= batch.stop:
            reference = pending.pop(0)
            for slide, row in enumerate(shifts):
                kept[slide].append(
                    score_slide(
                        network,
                        windows,
                        stored,
                        reference,
                        np.concatenate([[0], row]),
                        keep_threshold,
                    )
                )

            oldest = pending[0].start if pending else reference.stop
            for chunks in stored:
                while chunks and chunks[0][0] + len(chunks[0][1]) <= oldest:
                    chunks.pop(0)

    return [concatenate(parts) for parts in kept]


def roll_background(
    network: Dict[str, object],
    release_dir: Path,
    run: str,
    shifts: np.ndarray,
    output_dir: Path,
    vetoes: Dict[str, List[Interval]],
    stride_samples: int = 204,
    keep_threshold: float = 0.0,
    batch_size: int = 4096,
    device: str = "cuda",
) -> None:
    """Score roll slides and write one trigger file per slide.

    Every slide pairs all zero-lag windows; triggers whose data touches
    bad time are flagged ``vetoed``, and a slide's background time is its
    pairs clear of bad time times the stride.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    release_dir : Path
        Strain release directory.
    run : str
        Observing run.
    shifts : numpy.ndarray
        Output of ``roll_shifts``.
    output_dir : Path
        Directory for the trigger files.
    vetoes : dict of str to list of tuple of float
        ``sage.data.noise.data_quality.bad_segments`` per detector.
    stride_samples : int
        Step between window starts in samples.
    keep_threshold : float
        Ranking statistic above which a window is kept.
    batch_size : int
        Maximum windows scored at once.
    device : str
        Device to score on.
    """
    detectors = network["detectors"]
    segments = {
        detector: load_segments(release_dir, run, detector) for detector in detectors
    }
    windows = analysis_windows(segments, network["window_samples"], stride_samples)

    slides = score_rolls(
        network,
        release_dir,
        run,
        windows,
        shifts,
        stride_samples,
        keep_threshold,
        batch_size,
        device,
    )
    for row, columns in zip(shifts, slides):
        columns["vetoed"] = flag_vetoed(network, columns, vetoes)

        # Each window paired with its partners under this roll.
        paired = {"gps": windows["gps"]}
        for detector, shift in zip(detectors, np.concatenate([[0], row])):
            paired[f"{detector}_gps"] = np.roll(windows[f"{detector}_gps"], -shift)

        attrs = {
            **network["provenance"],
            "release_dir": str(Path(release_dir).resolve()),
            "run": run,
            "detectors": detectors,
            "method": "roll",
            "shifts": row,
            "n_windows": len(windows["gps"]),
            "n_live_windows": int((~flag_vetoed(network, paired, vetoes)).sum()),
            "stride_samples": stride_samples,
            "sample_rate": segments[detectors[0]][0]["sample_rate"],
            "keep_threshold": keep_threshold,
        }
        write_triggers(output_dir / f"roll_{row[0]}.h5", columns, attrs)
