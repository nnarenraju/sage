"""Time-slide background by random pairing of windows, as in sgwc-1.

Every detector gets uniformly random window starts in its own data; a
pair with any two detectors closer than a minimum time is redrawn.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from sage.data.noise.data_quality import Interval
from sage.search.dataprep.strain import scattered_window_batches
from sage.search.dataprep.windows import load_segments
from sage.search.model.score import score
from sage.search.removal.vetoes import flag_vetoed
from sage.search.zerolag.triggers import concatenate, loud_triggers, write_triggers


def random_starts(
    records: List[dict],
    n_windows: int,
    window_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly random window starts in one detector's data.

    Every start sample from which a whole window fits in a segment is
    equally likely.

    Parameters
    ----------
    records : list of dict
        Sidecar records of the detector.
    n_windows : int
        Number of windows to draw.
    window_samples : int
        Window length in samples.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    segment, sample : numpy.ndarray
        Segment index and start sample of each window.
    gps : numpy.ndarray
        Start time of each window.
    """
    sizes = [max(record["nsamples"] - window_samples + 1, 0) for record in records]
    edges = np.concatenate([[0], np.cumsum(sizes)])
    draws = rng.integers(0, edges[-1], n_windows)

    position = np.searchsorted(edges, draws, side="right") - 1
    sample = draws - edges[position]
    segment_starts = np.array([record["gps_start"] for record in records])
    segment_indices = np.array([record["segment_index"] for record in records])
    gps = segment_starts[position] + sample / records[0]["sample_rate"]
    return segment_indices[position], sample, gps


def random_pairs(
    segments: Dict[str, List[dict]],
    n_pairs: int,
    window_samples: int,
    min_separation_s: float = 100.0,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Random windows of every detector, paired, far enough apart.

    Each detector's starts are drawn independently. A pair with any two
    detectors closer than ``min_separation_s`` is redrawn in every
    detector, so the accepted pairs are uniform over the allowed ones.

    Parameters
    ----------
    segments : dict of str to list of dict
        Sidecar records per detector, the reference first.
    n_pairs : int
        Number of pairs.
    window_samples : int
        Window length in samples.
    min_separation_s : float
        Smallest time between any two windows of a pair.
    seed : int
        Seed of the random generator.

    Returns
    -------
    dict of str to numpy.ndarray
        ``gps`` of the reference window, and ``<detector>_segment``,
        ``<detector>_sample`` and ``<detector>_gps`` of every window.
    """
    rng = np.random.default_rng(seed)
    pairs = {}
    for detector in segments:
        pairs[f"{detector}_segment"] = np.zeros(n_pairs, dtype=np.int64)
        pairs[f"{detector}_sample"] = np.zeros(n_pairs, dtype=np.int64)
        pairs[f"{detector}_gps"] = np.zeros(n_pairs)

    redraw = np.arange(n_pairs)
    while redraw.size:
        for detector, records in segments.items():
            segment, sample, gps = random_starts(
                records, redraw.size, window_samples, rng
            )
            pairs[f"{detector}_segment"][redraw] = segment
            pairs[f"{detector}_sample"][redraw] = sample
            pairs[f"{detector}_gps"][redraw] = gps

        times = [pairs[f"{detector}_gps"][redraw] for detector in segments]
        too_close = np.zeros(redraw.size, dtype=bool)
        for first in range(len(times)):
            for second in range(first + 1, len(times)):
                too_close |= np.abs(times[first] - times[second]) < min_separation_s
        redraw = redraw[too_close]

    pairs["gps"] = pairs[f"{next(iter(segments))}_gps"].copy()
    return pairs


def score_pairs(
    network: Dict[str, object],
    release_dir: Path,
    run: str,
    pairs: Dict[str, np.ndarray],
    keep_threshold: float = 0.0,
    batch_size: int = 4096,
    device: str = "cuda",
    workers: int = 8,
) -> Dict[str, np.ndarray]:
    """Score random pairs with the full network.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    release_dir : Path
        Strain release directory.
    run : str
        Observing run.
    pairs : dict of str to numpy.ndarray
        Output of ``random_pairs``.
    keep_threshold : float
        Ranking statistic above which a pair is kept.
    batch_size : int
        Maximum pairs scored at once.
    device : str
        Device to score on.
    workers : int
        Processes reading strain.

    Returns
    -------
    dict of str to numpy.ndarray
        Trigger columns of the kept pairs.
    """
    kept = []
    for batch, strain in scattered_window_batches(
        release_dir,
        run,
        pairs,
        network["detectors"],
        network["window_samples"],
        batch_size,
        workers,
    ):
        block = strain.to(device=device, non_blocking=True)
        spectra = torch.fft.rfft(block, dim=-1, norm="forward")
        statistic, estimates = score(network, spectra)

        scored = {name: values[batch] for name, values in pairs.items()}
        kept.append(
            loud_triggers(network, statistic, estimates, scored, keep_threshold)
        )
    return concatenate(kept)


def random_pair_background(
    network: Dict[str, object],
    release_dir: Path,
    run: str,
    output: Path,
    n_pairs: int,
    vetoes: Dict[str, List[Interval]],
    min_separation_s: float = 100.0,
    seed: int = 0,
    stride_samples: int = 204,
    keep_threshold: float = 0.0,
    batch_size: int = 4096,
    device: str = "cuda",
    workers: int = 8,
) -> None:
    """Score random pairs and write their triggers to one file.

    Triggers whose data touches bad time are flagged ``vetoed``; the
    background time is the pairs clear of bad time times the stride, as
    in sgwc-1.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    release_dir : Path
        Strain release directory.
    run : str
        Observing run.
    output : Path
        Trigger file to write.
    n_pairs : int
        Number of random pairs.
    vetoes : dict of str to list of tuple of float
        ``sage.data.noise.data_quality.bad_segments`` per detector.
    min_separation_s : float
        Smallest time between any two windows of a pair.
    seed : int
        Seed of the random pairing.
    stride_samples : int
        Stride counted as each pair's share of background time.
    keep_threshold : float
        Ranking statistic above which a pair is kept.
    batch_size : int
        Maximum pairs scored at once.
    device : str
        Device to score on.
    workers : int
        Processes reading strain.
    """
    detectors = network["detectors"]
    segments = {
        detector: load_segments(release_dir, run, detector) for detector in detectors
    }
    pairs = random_pairs(
        segments, n_pairs, network["window_samples"], min_separation_s, seed
    )

    columns = score_pairs(
        network,
        release_dir,
        run,
        pairs,
        keep_threshold,
        batch_size,
        device,
        workers,
    )
    columns["vetoed"] = flag_vetoed(network, columns, vetoes)

    attrs = {
        **network["provenance"],
        "release_dir": str(Path(release_dir).resolve()),
        "run": run,
        "detectors": detectors,
        "method": "random_pairs",
        "n_windows": n_pairs,
        "n_live_windows": int((~flag_vetoed(network, pairs, vetoes)).sum()),
        "min_separation_s": min_separation_s,
        "seed": seed,
        "stride_samples": stride_samples,
        "sample_rate": segments[detectors[0]][0]["sample_rate"],
        "keep_threshold": keep_threshold,
    }
    write_triggers(output, columns, attrs)
