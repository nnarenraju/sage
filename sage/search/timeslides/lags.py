"""Time-slide background by fixed time lags, as in PyCBC.

Each other detector's data is read a whole number of lag steps after
the reference detector's; the windows are rebuilt for every lag.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np

from sage.data.noise.data_quality import Interval
from sage.search.dataprep.windows import analysis_windows, load_segments
from sage.search.removal.vetoes import flag_vetoed
from sage.search.zerolag.triggers import write_triggers
from sage.search.zerolag.zerolag import score_windows


def time_lags(
    n_slides: int, n_followers: int, min_separation_s: float = 100.0
) -> np.ndarray:
    """Lag of each other detector for each slide.

    Slide ``k`` lags detector ``f`` by ``k (f + 1)`` times the minimum
    separation, so every detector pair is at least that far apart.

    Parameters
    ----------
    n_slides : int
        Number of slides.
    n_followers : int
        Number of detectors other than the reference.
    min_separation_s : float
        Lag step and smallest time between paired windows.

    Returns
    -------
    numpy.ndarray
        Lags in seconds, shape ``(n_slides, n_followers)``.
    """
    slides = np.arange(1, n_slides + 1)[:, None]
    return min_separation_s * slides * np.arange(1, n_followers + 1)


def lag_background(
    network: Dict[str, object],
    release_dir: Path,
    run: str,
    lags: np.ndarray,
    output_dir: Path,
    vetoes: Dict[str, List[Interval]],
    stride_samples: int = 204,
    keep_threshold: float = 0.0,
    batch_size: int = 4096,
    device: str = "cuda",
) -> None:
    """Score each lag and write one trigger file per lag.

    Triggers whose data touches bad time are flagged ``vetoed``; a
    slide's background time is its windows clear of bad time times the
    stride, stored with its triggers.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    release_dir : Path
        Strain release directory.
    run : str
        Observing run.
    lags : numpy.ndarray
        Output of ``time_lags``.
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

    for row in lags:
        windows = analysis_windows(
            segments,
            network["window_samples"],
            stride_samples,
            lags=dict(zip(detectors[1:], row.tolist())),
        )
        columns = score_windows(
            network,
            release_dir,
            run,
            windows,
            stride_samples,
            keep_threshold,
            batch_size,
            device,
        )
        columns["vetoed"] = flag_vetoed(network, columns, vetoes)

        attrs = {
            **network["provenance"],
            "release_dir": str(Path(release_dir).resolve()),
            "run": run,
            "detectors": detectors,
            "method": "lag",
            "lags_s": row,
            "n_windows": len(windows["gps"]),
            "n_live_windows": int((~flag_vetoed(network, windows, vetoes)).sum()),
            "stride_samples": stride_samples,
            "sample_rate": segments[detectors[0]][0]["sample_rate"],
            "keep_threshold": keep_threshold,
        }
        write_triggers(output_dir / f"lag_{round(row[0])}.h5", columns, attrs)
