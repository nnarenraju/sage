"""Read fiducial ASDs and check them against the run.

Files are ``fiducial_<det>_psd.bin`` (float32 ASDs, strain/sqrt(Hz)) with a
JSON sidecar describing the grid they were built on. A checkpoint carries the
same pairs in its record.
"""

import json
from pathlib import Path

import numpy as np
import torch

from sage.core.geometry import Window


def check_fiducial(meta: dict, asd: np.ndarray, detector: str, window: Window, where: str) -> None:
    """
    Check one fiducial ASD and its sidecar against the run.

    Parameters
    ----------
    meta : dict
        Sidecar contents.
    asd : numpy.ndarray
        ASD values.
    detector : str
        Detector the ASD must belong to.
    window : Window
        Window geometry the ASD must match.
    where : str
        Location, for error messages.

    Raises
    ------
    ValueError
        If the detector, sample rate, frequency spacing, bin count, dtype or
        byte order does not match.
    """
    expected = {
        "dtype": "float32",
        "byte_order": "little",
        "sample_rate": window.sample_rate_in_hz,
        "delta_f": window.delta_f_in_hz,
        "num_freq_bins": window.n_frequency_bins,
        "detector": detector,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{where}: {key} is {meta.get(key)!r}, run needs {value!r}")
    if asd.size != window.n_frequency_bins:
        raise ValueError(f"{where}: {asd.size} bins, run needs {window.n_frequency_bins}")


def read_fiducial_asds(
    fiducial_dir: Path, detectors: list, window: Window, device
) -> torch.Tensor:
    """
    Read one fiducial ASD per detector from a directory, checking each.

    Parameters
    ----------
    fiducial_dir : Path
        Directory holding the fiducial files.
    detectors : list of str
        Detectors, in channel order.
    window : Window
        Window geometry the ASDs must match.
    device : str or torch.device
        Device of the returned tensor.

    Returns
    -------
    torch.Tensor, shape ``(D, F)``
        ASDs per detector.

    Raises
    ------
    FileNotFoundError
        If a detector's file is missing.
    ValueError
        If a file does not match the run (see :func:`check_fiducial`).
    """
    asds = []
    for detector in detectors:
        meta_path = Path(fiducial_dir) / f"fiducial_{detector}_psd.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"fiducial ASD for {detector} missing: {meta_path}")
        asd = np.fromfile(meta_path.with_suffix(".bin"), dtype=np.float32)
        check_fiducial(json.loads(meta_path.read_text()), asd, detector, window, str(meta_path))
        asds.append(asd)
    return torch.from_numpy(np.stack(asds, axis=0)).to(device=device)


def fiducial_asds_from_record(
    files: dict, detectors: list, window: Window, device
) -> torch.Tensor:
    """
    The fiducial ASDs a checkpoint record carries, checked like files.

    Parameters
    ----------
    files : dict of str to bytes
        ``record["fiducial"]``: file name to contents.
    detectors : list of str
        Detectors, in channel order.
    window : Window
        Window geometry the ASDs must match.
    device : str or torch.device
        Device of the returned tensor.

    Returns
    -------
    torch.Tensor, shape ``(D, F)``
        ASDs per detector.

    Raises
    ------
    ValueError
        If a detector is missing or does not match the run.
    """
    asds = []
    for detector in detectors:
        name = f"fiducial_{detector}_psd"
        if f"{name}.json" not in files:
            raise ValueError(f"checkpoint record has no fiducial ASD for {detector}")
        asd = np.frombuffer(files[f"{name}.bin"], dtype=np.float32)
        check_fiducial(json.loads(files[f"{name}.json"]), asd, detector, window, f"record {name}")
        asds.append(asd)
    return torch.from_numpy(np.stack(asds, axis=0)).to(device=device)
