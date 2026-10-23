"""Score every zero-lag window and keep the loud ones as triggers."""

from pathlib import Path
from typing import Dict, List

import numpy as np

from sage.data.noise.data_quality import Interval
from sage.search.dataprep.strain import window_batches
from sage.search.dataprep.windows import analysis_windows, load_segments
from sage.search.model.score import score_strain
from sage.search.removal.vetoes import flag_vetoed
from sage.search.zerolag.triggers import concatenate, loud_triggers, write_triggers


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
    """Score analysis windows and return the loud ones as trigger columns.

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


def zerolag(
    network: Dict[str, object],
    release_dir: Path,
    run: str,
    output: Path,
    vetoes: Dict[str, List[Interval]],
    stride_samples: int = 204,
    keep_threshold: float = 0.0,
    batch_size: int = 4096,
    device: str = "cuda",
) -> None:
    """Score every zero-lag window and write the loud ones to one file.

    Triggers whose data touches bad time are flagged ``vetoed``; the
    analysed time is ``n_live_windows * stride_samples / sample_rate``.

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
    vetoes : dict of str to list of tuple of float
        ``sage.data.noise.data_quality.bad_segments`` per detector.
    stride_samples : int
        Step between window starts in samples.
    keep_threshold : float
        Ranking statistic above which a window is kept as a trigger.
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
        "n_windows": len(windows["gps"]),
        "n_live_windows": int((~flag_vetoed(network, windows, vetoes)).sum()),
        "stride_samples": stride_samples,
        "sample_rate": segments[detectors[0]][0]["sample_rate"],
        "keep_threshold": keep_threshold,
    }
    write_triggers(output, columns, attrs)
