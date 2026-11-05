"""
MLGWSC-1 benchmark of a stored checkpoint of a run, on dataset 4.

Slicing and clustering follow the MLGWSC-1 example search; every window
goes through the run's own preprocessing (``compare.load_network``).
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from sage.benchmark.mlgwsc1 import release
from sage.benchmark.mlgwsc1.evaluator import evaluate
from sage.config.machine import read_data_dir
from sage.core.logger import get_logger
from sage.factory.compare import load_network, scores
from sage.factory.train_run import check_run, chosen_checkpoint

logger = get_logger(__name__)

STEP_IN_S = 0.1  # between the starts of consecutive windows
TRIGGER_THRESHOLD = 0.0  # ranking statistic above which a window is a trigger
CLUSTER_IN_S = 0.35  # triggers at most this far apart form one event
TIME_TOLERANCE_IN_S = 0.35  # each event's var: how far an injection may lie
BATCH_SIZE = 256
WORKERS = 8


class Slicer(torch.utils.data.Dataset):
    """
    Overlapping padded windows of an MLGWSC-1 data file, in time order.

    Parameters
    ----------
    path : Path
        MLGWSC-1 HDF5 file: a group per detector, a dataset per segment.
    detectors : list of str
        The run's detectors, in its order.
    window : Window
        The run's window geometry.
    peak_offset : float
        Seconds from a window's start to its expected merger.

    Raises
    ------
    ValueError
        If the file holds other detectors or another sample rate.
    """

    def __init__(self, path: Path, detectors: list, window, peak_offset: float):
        self.path, self.detectors, self.peak_offset = path, detectors, peak_offset
        self.rate = window.sample_rate_in_hz
        self.length = window.padded_length_in_samples
        self.step = int(STEP_IN_S * self.rate)
        self.file = None  # opened in each worker

        with h5py.File(path, "r") as file:
            found = sorted(str(detector) for detector in file.attrs["detectors"])
            if found != sorted(detectors):
                raise ValueError(f"{path}: detectors {found}, the run has {detectors}")
            self.keys = sorted(file[detectors[0]].keys(), key=int)
            segments = [file[detectors[0]][key] for key in self.keys]
            for key, segment in zip(self.keys, segments):
                if segment.attrs["delta_t"] != 1 / self.rate:
                    raise ValueError(
                        f"{path}: segment {key} has delta_t "
                        f"{segment.attrs['delta_t']}, the run 1/{self.rate}"
                    )
            counts = [max((len(ds) - self.length) // self.step, 0) for ds in segments]
            self.starts = [float(segment.attrs["start_time"]) for segment in segments]

        self.first = np.cumsum([0, *counts])  # first window of each segment

    def __len__(self) -> int:
        return int(self.first[-1])

    def __getitem__(self, index: int) -> tuple:
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        segment = int(np.searchsorted(self.first, index, side="right")) - 1
        start = (index - int(self.first[segment])) * self.step
        key = self.keys[segment]
        strain = np.stack([self.file[detector][key][start : start + self.length]
                           for detector in self.detectors]).astype(np.float32)
        time = self.starts[segment] + start / self.rate + self.peak_offset
        return torch.from_numpy(strain), torch.tensor(time, dtype=torch.float64)


def peak_offset(run: SimpleNamespace) -> float:
    """Seconds from a window's start to the middle of the delta_tc prior."""
    low, high = run.bounds["delta_tc"]
    return run.window.padding_length_in_s + (low + high) / 2


def triggers(run: SimpleNamespace, path: Path, device: torch.device) -> np.ndarray:
    """
    Windows of ``path`` whose ranking statistic exceeds the threshold.

    Parameters
    ----------
    run : SimpleNamespace
        A loaded run (``compare.load_network``).
    path : Path
        MLGWSC-1 data file.
    device : torch.device
        Device the run is loaded on.

    Returns
    -------
    numpy.ndarray, shape (N, 2)
        Expected merger time (GPS) and ranking statistic, in time order.

    Raises
    ------
    ValueError
        If no window exceeds the threshold.
    """
    slicer = Slicer(path, run.detectors, run.window, peak_offset(run))
    loader = torch.utils.data.DataLoader(
        slicer, batch_size=BATCH_SIZE, num_workers=WORKERS, prefetch_factor=4,
        pin_memory=device.type == "cuda", persistent_workers=True,
    )
    found, loudest = [], -np.inf
    with torch.inference_mode():
        for strain, times in loader:
            strain = strain.to(device=device, dtype=run.dtype, non_blocking=True)
            rank = scores(run, torch.fft.rfft(strain, dim=-1, norm="forward"))[:, 0]
            above = rank > TRIGGER_THRESHOLD
            found.append(np.stack([times.numpy()[above], rank[above]], axis=1))
            loudest = max(loudest, float(rank.max()))

    found = np.concatenate(found)
    logger.info("%s: %d windows, %d above %s, loudest %.4f",
                path.name, len(slicer), len(found), TRIGGER_THRESHOLD, loudest)
    if len(found) == 0:
        raise ValueError(
            f"{path}: no window above the trigger threshold {TRIGGER_THRESHOLD}"
        )
    return found


def clusters(found: np.ndarray) -> tuple:
    """
    The loudest trigger of each run of triggers at most CLUSTER_IN_S apart.

    Parameters
    ----------
    found : numpy.ndarray, shape (N, 2)
        Triggers in time order (:func:`triggers`).

    Returns
    -------
    tuple of numpy.ndarray
        Event times, ranking statistics and time tolerances.
    """
    breaks = np.flatnonzero(np.diff(found[:, 0]) > CLUSTER_IN_S) + 1
    events = np.array([part[np.argmax(part[:, 1])] for part in np.split(found, breaks)])
    return events[:, 0], events[:, 1], np.full(len(events), TIME_TOLERANCE_IN_S)


def search(run: SimpleNamespace, path: Path, output: Path, device: torch.device) -> int:
    """
    Search ``path`` and write its events to ``output``.

    Parameters
    ----------
    run : SimpleNamespace
        A loaded run (``compare.load_network``).
    path : Path
        MLGWSC-1 data file.
    output : Path
        HDF5 file with the datasets ``time``, ``stat`` and ``var``,
        written through ``<output>.part``.
    device : torch.device
        Device the run is loaded on.

    Returns
    -------
    int
        Number of events.
    """
    time, stat, var = clusters(triggers(run, path, device))
    part = output.with_name(output.name + ".part")
    with h5py.File(part, "w") as file:
        file.create_dataset("time", data=time)
        file.create_dataset("stat", data=stat)
        file.create_dataset("var", data=var)
    part.rename(output)
    return len(time)


def benchmark(
    run_dir: Path, from_epoch: int | None, ema_calibrated: bool, workers: int
) -> Path:
    """
    Benchmark one stored checkpoint of a run on MLGWSC-1 dataset 4.

    MLGWSC-1's code, every team's results, dataset 4 and its injection
    SNRs are made once in ``<data_dir>/mlgwsc1``. The events and the
    evaluation go into ``<run>/results/mlgwsc1/<checkpoint>``, built in
    a ``.part`` folder; finished event files are kept if a job stops.

    Parameters
    ----------
    run_dir : Path
        A started run directory.
    from_epoch : int or None
        Stored epoch; None takes ``best.pt``.
    ema_calibrated : bool
        Take ``ema_calibrated.pt`` instead.
    workers : int
        Processes for the injection SNRs.

    Returns
    -------
    Path
        The benchmark's folder.

    Raises
    ------
    FileExistsError
        If this checkpoint is already benchmarked.
    """
    run_dir = Path(run_dir)
    checkpoint, name = chosen_checkpoint(run_dir, from_epoch, ema_calibrated)
    output = run_dir / "results" / "mlgwsc1" / name
    if output.exists():
        raise FileExistsError(f"{output} exists: {name} is already benchmarked")

    data_dir = read_data_dir()
    directory = data_dir / "mlgwsc1"
    directory.mkdir(exist_ok=True)
    code = release.fetch_code(directory)
    results = release.fetch_results(directory)
    dataset = release.make_dataset(directory, code)
    release.make_snrs(dataset, code, workers)

    settings = check_run(run_dir)[0]
    device = torch.device(settings["train"]["device"])
    run = load_network(run_dir, checkpoint, device, data_dir)
    part = output.with_name(output.name + ".part")
    part.mkdir(parents=True, exist_ok=True)
    for kind, data in (("bg", "background.hdf"), ("fg", "foreground.hdf")):
        events = part / f"{kind}_events.hdf"
        if not events.exists():
            logger.info("Searching %s", dataset / data)
            search(run, dataset / data, events, device)

    evaluation = part / "evaluation"
    shutil.rmtree(evaluation, ignore_errors=True)  # an evaluation cut short
    evaluation.mkdir()
    evaluate(str(dataset), f"{run.name} ({name})", str(part / "fg_events.hdf"),
             str(part / "bg_events.hdf"), str(results), str(evaluation))
    part.rename(output)
    return output
