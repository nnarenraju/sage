"""
Optimal network SNR of each MLGWSC-1 injection, kept beside the dataset.

Each injection's projected signal is the one MLGWSC-1 adds (PyCBC
``InjectionSet``); its ``sigmasq`` is taken against the per-frequency
median of MLGWSC-1's PSDs of each detector.
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pycbc.filter import sigmasq
from pycbc.inject import InjectionSet
from pycbc.psd import interpolate
from pycbc.types import FrequencySeries, load_frequencyseries

DETECTORS = ("H1", "L1")  # MLGWSC-1 data sets
SAMPLE_RATE = 2048.0  # MLGWSC-1 data sets
F_LOWER = 20.0  # MLGWSC-1 injections and generate_data.make_injections
CHUNK = 1000  # injections per task


def median_psds(psd_dir: Path) -> dict:
    """
    Per-frequency median of MLGWSC-1's PSDs of each detector.

    Parameters
    ----------
    psd_dir : Path
        MLGWSC-1's ``psds`` directory: ``<detector>/psd-<i>.hdf``.

    Returns
    -------
    dict of str to FrequencySeries
        Median PSD per detector.

    Raises
    ------
    ValueError
        If a detector has no PSDs, or its PSDs differ in grid.
    """
    medians = {}
    for detector in DETECTORS:
        paths = sorted((psd_dir / detector).glob("psd-*.hdf"))
        psds = [load_frequencyseries(str(path)) for path in paths]
        if not psds:
            raise ValueError(f"{psd_dir / detector}: no psd-*.hdf files")
        grids = {(psd.delta_f, len(psd)) for psd in psds}
        if len(grids) > 1:
            raise ValueError(
                f"{psd_dir / detector}: PSDs on different grids {sorted(grids)}"
            )
        median = np.median(np.stack([psd.numpy() for psd in psds]), axis=0)
        medians[detector] = FrequencySeries(median, delta_f=psds[0].delta_f)
    return medians


def chunk_snrs(path: Path, indices: np.ndarray, psds: dict) -> np.ndarray:
    """
    Optimal network SNR of the injections ``indices`` of ``path``.

    Parameters
    ----------
    path : Path
        MLGWSC-1 injection file.
    indices : numpy.ndarray of int
        Injections to measure.
    psds : dict of str to FrequencySeries
        PSD per detector (:func:`median_psds`).

    Returns
    -------
    numpy.ndarray
        Network SNR of each injection.
    """
    injections = InjectionSet(str(path))
    snrs = np.empty(len(indices))
    for row, index in enumerate(indices):
        total = 0.0
        for detector in DETECTORS:
            signal = injections.make_strain_from_inj_object(
                injections.table[index], 1 / SAMPLE_RATE, detector, f_lower=F_LOWER
            )
            if len(signal) % 2:
                signal.append_zeros(1)
            psd = interpolate(psds[detector], 1 / signal.duration, len(signal) // 2 + 1)
            total += sigmasq(signal, psd=psd, low_frequency_cutoff=F_LOWER)
        snrs[row] = np.sqrt(total)
    return snrs


def injection_snrs(
    path: Path, psd_dir: Path, inside: np.ndarray, workers: int
) -> np.ndarray:
    """
    Optimal network SNR of the injections in the data, in file order.

    Parameters
    ----------
    path : Path
        MLGWSC-1 injection file.
    psd_dir : Path
        MLGWSC-1's ``psds`` directory.
    inside : numpy.ndarray of bool
        Injections in the foreground data (``find_injection_times``).
    workers : int
        Worker processes.

    Returns
    -------
    numpy.ndarray
        Network SNR of each injection; NaN for those not in the data.
    """
    psds = median_psds(psd_dir)
    indices = np.flatnonzero(inside)
    chunks = np.array_split(indices, max(1, -(-len(indices) // CHUNK)))
    snrs = np.full(len(inside), np.nan)
    with Pool(workers) as pool:
        tasks = [(path, chunk, psds) for chunk in chunks]
        snrs[indices] = np.concatenate(pool.starmap(chunk_snrs, tasks))
    return snrs
