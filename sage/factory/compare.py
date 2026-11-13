"""
Score several trained runs on identical validation windows.

For ``sage compare``: each batch is drawn once from the first run's
validation streams and scored by every run's ``best.pt``.
"""

import dataclasses
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml

from sage.architecture.custom_losses import width_to_sigma
from sage.config.machine import fiducial_dir, read_data_dir
from sage.config.run_dir import check_record, differences, record_fiducial, training_started
from sage.core.logger import get_logger
from sage.core.pipeline import GWBatch, Grid, ProcessingState
from sage.data.asd import read_fiducial_asds
from sage.data.waveform.sampler import prior_sampler
from sage.factory.build import (
    build_data,
    build_loss,
    build_model,
    build_noise_stream,
    build_preprocess,
    build_signal,
    check_device,
    precision_setup,
    resolve_settings,
)
from sage.factory.train_run import check_data, check_run

logger = get_logger(__name__)

OUTPUT_FILE = "compare.h5"
REPORT_FILE = "report.txt"
INCOMPLETE_FILE = "incomplete"  # removed once compare.h5 is complete
FALSE_ALARM_PROBABILITIES = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
SNR_TABLE_PROBABILITIES = (1e-5, 1e-6)
SNR_EDGES = (5, 6, 7, 8, 9, 10, 12, 15, np.inf)
FLUSH_EVERY = 200


class WindowMismatch(ValueError):
    """A run does not share the first run's detectors and window."""


class Appender:
    """
    A resizable HDF5 dataset, appended along axis 0.

    Parameters
    ----------
    group : h5py.Group
        Group the dataset is created in.
    name : str
        Dataset name.
    width : int
        Columns per row.
    dtype : numpy dtype
        Dataset dtype.
    """

    def __init__(self, group, name, width, dtype=np.float32):
        self.ds = group.create_dataset(
            name, shape=(0, width), maxshape=(None, width), dtype=dtype,
            chunks=(65536, width), compression="gzip",
        )
        self.buf = []

    def add(self, arr):
        """Buffer rows ``arr`` until the next ``flush``."""
        self.buf.append(np.asarray(arr))

    def flush(self):
        """Append the buffered rows to the dataset."""
        if not self.buf:
            return
        data = np.concatenate(self.buf)
        n = self.ds.shape[0]
        self.ds.resize(n + len(data), axis=0)
        self.ds[n:] = data
        self.buf = []


def load_network(run_dir: Path, checkpoint: Path, device, data_dir: Path) -> SimpleNamespace:
    """
    A run's checkpoint with its preprocessing, scaling and precision.

    Parameters
    ----------
    run_dir : Path
        A started run directory.
    checkpoint : Path
        One of its stored checkpoints (``chosen_checkpoint``).
    device : torch.device
        Device every network runs on.
    data_dir : Path
        The machine's data directory.

    Returns
    -------
    SimpleNamespace
        ``path``, ``name``, ``epoch``, ``window``, ``detectors``,
        ``dtype``, ``bounds`` (prior bounds), ``asds``, ``processor``,
        ``network``, ``loss``, ``targets``, ``offset``, ``scale``,
        ``autocast_dtype`` and ``columns`` (output column names).

    Raises
    ------
    ValueError
        On invalid settings or a device not available here.
    FileNotFoundError
        If the run has no checkpoint.
    SettingsChanged
        If the settings or prior changed after training started.
    """
    settings, prior = check_run(run_dir)
    if not training_started(run_dir):
        raise FileNotFoundError(f"{run_dir}: no checkpoint to compare")
    check_record(run_dir, resolve_settings(settings), prior, resolve_settings)
    train = {**settings["train"], "device": str(device)}
    check_device(train)
    window, detectors, dtype = build_data(settings["data"])
    targets = list(settings["signal"]["targets"])
    fiducial = record_fiducial(run_dir, fiducial_dir(data_dir, settings["noise"]["fiducial"]), detectors)
    asds = read_fiducial_asds(fiducial, detectors, window, device)
    bounds = prior_sampler(prior, settings["validation"]["seed"], device, dtype).bounds
    processor = build_preprocess(
        settings["preprocess"], asds, window, bounds, settings["signal"]["low_frequency_cutoff_in_hz"],
        ProcessingState(Grid.FD_UNIFORM), device, dtype,
    )
    network = build_model(
        settings["model"], len(detectors), len(targets), "non_astrophysical" in settings["signal"], device, dtype
    )
    network = network.to(memory_format=torch.channels_last)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    network.load_state_dict(state["model"])
    network.eval()
    scaling = yaml.safe_load((run_dir / "resolved" / "derived.yaml").read_text())["target_scaling"]
    loss = build_loss(settings["loss"], targets)
    return SimpleNamespace(
        path=run_dir, name=run_dir.resolve().name, epoch=int(state["epoch"]), window=window, detectors=detectors,
        dtype=dtype, bounds=bounds,
        asds=asds, processor=processor, network=network, loss=loss, targets=targets,
        offset=torch.tensor([scaling[target]["offset"] for target in targets], device=device),
        scale=torch.tensor([scaling[target]["scale"] for target in targets], device=device),
        autocast_dtype=precision_setup(train)[0],
        columns=["rank", *(f"mu_{target}" for target in targets), *(f"sigma_{target}" for target in targets)],
    )


def scores(run: SimpleNamespace, x: torch.Tensor) -> np.ndarray:
    """
    Rank, then means and widths in physical units, for a batch.

    Parameters
    ----------
    run : SimpleNamespace
        Run from ``load_network``.
    x : torch.Tensor
        Batch of windows, before preprocessing.

    Returns
    -------
    numpy.ndarray
        One row per window: rank, means and, if the model has them,
        widths (sigma).
    """
    batch = run.processor(GWBatch(x, state=run.processor.input_state))
    with (torch.autocast(device_type=x.device.type, dtype=run.autocast_dtype)
          if run.autocast_dtype is not None else nullcontext()):
        out = run.network(batch.to_network_input())
    # Rank and estimates; not the non-astrophysical logit
    raw = torch.cat([out[0], out[1]], dim=1).float()
    count = len(run.targets)
    columns = [raw[:, :1], raw[:, 1:1 + count] * run.scale + run.offset]
    if raw.shape[1] >= 1 + 2 * count:  # mean-only models have no widths
        columns.append(width_to_sigma(raw[:, 1 + count:1 + 2 * count], run.loss.sigma_min, run.loss.sigma_max) * run.scale)
    return torch.cat(columns, dim=1).cpu().numpy()


def compare_runs(run_dirs: list, output_dir: Path, noise_windows: int, signal_windows: int, seed: int) -> str:
    """
    Score every run's ``best.pt`` on the same windows.

    The windows follow the first run's validation noise, prior and SNR
    distribution, seeded from ``seed`` alone. Scores are appended to
    ``compare.h5``; ``incomplete`` sits beside it until it is complete.

    Parameters
    ----------
    run_dirs : list of Path
        Started run directories with distinct names; all share the
        first one's detectors and window geometry.
    output_dir : Path
        New directory for ``compare.h5``, ``report.txt`` and the
        figures (``plot_comparison``); it must not exist.
    noise_windows, signal_windows : int
        Noise-only windows and windows carrying a signal (whole
        batches: at least these many).
    seed : int
        Seed of the windows.

    Returns
    -------
    str
        The report (:func:`report`).

    Raises
    ------
    ValueError
        If two runs share a directory name, a window count is below
        one or a run's settings are invalid.
    WindowMismatch
        If a run does not share the first one's detectors and window
        geometry.
    FileExistsError
        If ``output_dir`` exists.
    """
    if noise_windows < 1 or signal_windows < 1:
        raise ValueError(f"windows: {noise_windows} noise, {signal_windows} signal; give at least one of each")
    run_dirs = [Path(run_dir) for run_dir in run_dirs]
    names = [run_dir.resolve().name for run_dir in run_dirs]
    if len(set(names)) < len(names):
        raise ValueError(f"runs are named by their directories, and these repeat: {', '.join(names)}")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} exists; every comparison writes into a new directory")
    path = output_dir / OUTPUT_FILE
    data_dir = read_data_dir()
    reference, prior = check_run(run_dirs[0])
    check_data(run_dirs[0], reference, data_dir)
    device = torch.device(reference["train"]["device"])
    runs = [load_network(run_dir, run_dir / "checkpoints" / "best.pt", device, data_dir) for run_dir in run_dirs]
    first = runs[0]
    for run in runs[1:]:
        found = differences(
            {"detectors": first.detectors, "window": dataclasses.asdict(first.window)},
            {"detectors": run.detectors, "window": dataclasses.asdict(run.window)},
        )
        if found:
            raise WindowMismatch(f"{run.path} does not share the windows of {first.path}: " + "; ".join(found))

    train, validation, replay = reference["train"], reference["validation"], reference["mining"]["replay"]
    _, _, dtype = build_data(reference["data"])
    signal = build_signal(
        {**reference["signal"], "snr": validation["snr"]}, prior, first.window, first.detectors, first.asds,
        round(train["batch_size"] * train["signal_fraction"]), seed, 0, device, dtype,
    )
    noise = build_noise_stream(
        reference["noise"], "validation", data_dir, first.window, first.detectors, train["batch_size"], seed, 0,
        replay["start_jitter_in_s"], replay["focal_gamma"], device, dtype,
    )
    output_dir.mkdir(parents=True)
    incomplete = output_dir / INCOMPLETE_FILE
    incomplete.write_text(f"{OUTPUT_FILE} holds only the windows scored before this comparison stopped\n")
    try:
        with h5py.File(path, "w") as file, torch.inference_mode():
            file.attrs.update(
                seed=seed, runs=names, windows_from=str(first.path), complete=False,
                snr=", ".join(f"{key} {value}" for key, value in validation["snr"].items()),
            )
            for run in runs:
                file.attrs[f"{run.name}_checkpoint"] = str(run.path / "checkpoints" / "best.pt")
                file.attrs[f"{run.name}_epoch"] = run.epoch
            background, foreground = file.create_group("background"), file.create_group("foreground")
            outputs = {}

            def add(group, run, x):
                """Buffer one run's scores of ``x`` in ``group``."""
                values = scores(run, x)
                # Mean-only models have no width columns
                if (group.name, run.name) not in outputs:
                    outputs[group.name, run.name] = Appender(group, run.name, values.shape[1])
                    group[run.name].attrs["columns"] = run.columns[:values.shape[1]]
                outputs[group.name, run.name].add(values)

            targets = parameters = snrs = None

            # Foreground: every window of a batch carries a signal.
            done = iteration = 0
            while done < signal_windows:
                noise_x = noise()[0]
                parts = []
                while sum(len(part[0]) for part in parts) < len(noise_x):
                    strain, target, theta = signal(return_theta=True)
                    parts.append((strain, target, theta, signal.augment.target_snr_sampler.last))
                strain, target, theta, snr = (torch.cat([part[k] for part in parts])[:len(noise_x)] for k in range(4))
                if targets is None:
                    targets = Appender(foreground, "network_target", target.shape[1])
                    parameters = Appender(foreground, "signal_params", theta.shape[1])
                    foreground["signal_params"].attrs["names"] = signal.param_sampler.param_names
                    snrs = Appender(foreground, "target_snr", 1)
                x = noise_x + strain
                for run in runs:
                    add(foreground, run, x)
                targets.add(target.cpu().numpy())
                parameters.add(theta.cpu().numpy())
                snrs.add(snr.cpu().numpy()[:, None])
                done += len(noise_x)
                iteration += 1
                if iteration % FLUSH_EVERY == 0:
                    for appender in (*outputs.values(), targets, parameters, snrs):
                        appender.flush()
                    file.flush()
                    logger.info("Foreground %d of %d windows", done, signal_windows)
            for appender in (*outputs.values(), targets, parameters, snrs):
                appender.flush()

            # Background: noise-only windows.
            done = iteration = 0
            while done < noise_windows:
                noise_x = noise()[0]
                for run in runs:
                    add(background, run, noise_x)
                done += len(noise_x)
                iteration += 1
                if iteration % FLUSH_EVERY == 0:
                    for run in runs:
                        outputs[background.name, run.name].flush()
                    file.flush()
                    logger.info("Background %d of %d windows", done, noise_windows)
            for run in runs:
                outputs[background.name, run.name].flush()
            file.attrs["complete"] = True
        incomplete.unlink()
    finally:
        noise.shutdown()
    text = report(path)
    (path.parent / REPORT_FILE).write_text(text + "\n")

    # Imported here: the plotting libraries load only for a comparison
    from sage.plotting.comparison import plot_comparison

    plot_comparison(path, run_dirs, data_dir)
    return text


def recovered_fractions(background: np.ndarray, foreground: np.ndarray, probabilities) -> list:
    """
    Recovered signal fraction at fixed false-alarm probabilities.

    The threshold is the ``count``-th loudest background score, with
    ``count = round(probability * windows)``; the loudest when
    ``count`` is 0. A signal is recovered above the threshold.

    Parameters
    ----------
    background : numpy.ndarray
        Ranking statistic of the noise-only windows.
    foreground : numpy.ndarray
        Ranking statistic of the signal windows.
    probabilities : iterable of float
        False-alarm probabilities per window.

    Returns
    -------
    list of tuple
        ``(probability, count, threshold, fraction)`` per probability.
    """
    loudest = np.sort(background)[::-1]
    rows = []
    for probability in probabilities:
        count = int(round(probability * len(loudest)))
        threshold = loudest[count - 1] if count >= 1 else loudest[0]
        rows.append((probability, count, threshold, float(np.mean(foreground > threshold))))
    return rows


def report(path: Path) -> str:
    """
    The recovered-fraction tables and loudest noise windows of a file.

    Parameters
    ----------
    path : Path
        A complete ``compare.h5``.

    Returns
    -------
    str
        Report text.

    Raises
    ------
    ValueError
        If ``path`` is not complete.
    """
    with h5py.File(path, "r") as file:
        if not file.attrs["complete"]:
            raise ValueError(f"{path} is not complete")
        names = [str(name) for name in file.attrs["runs"]]
        background = {name: file["background"][name][:, 0].astype(np.float64) for name in names}
        foreground = {name: file["foreground"][name][:, 0].astype(np.float64) for name in names}
        snr = file["foreground/target_snr"][:, 0].astype(np.float64)
        snr_settings = file.attrs["snr"]
    fractions = {name: recovered_fractions(background[name], foreground[name], FALSE_ALARM_PROBABILITIES)
                 for name in names}
    width = max(22, *(len(name) + 2 for name in names))
    lines = [f"background windows {len(background[names[0]]):,}, foreground signals {len(snr):,} "
             f"(target network SNR {snr_settings})", "",
             "Recovered fraction at fixed noise false-alarm probability per window",
             "FAP      " + "".join(f"{name:>{width}s}" for name in names)]
    for row, probability in enumerate(FALSE_ALARM_PROBABILITIES):
        line = f"{probability:7.0e}  "
        for name in names:
            _, _, threshold, fraction = fractions[name][row]
            line += f"{f'{fraction:7.4f} (thr {threshold:6.2f})':>{width}s}"
        count = fractions[names[0]][row][1]
        lines.append(line + (f"  <- set by top {count} noise windows" if count <= 10 else ""))
    for probability in SNR_TABLE_PROBABILITIES:
        row = FALSE_ALARM_PROBABILITIES.index(probability)
        lines += ["", f"Recovered fraction by target SNR at FAP {probability:.0e}",
                  "SNR bin       n      " + "".join(f"{name:>{width}s}" for name in names)]
        for low, high in zip(SNR_EDGES[:-1], SNR_EDGES[1:]):
            mask = (snr >= low) & (snr < high)
            line = f"[{low:>4},{high:>4})  {mask.sum():7d}  "
            line += "".join(f"{np.mean(foreground[name][mask] > fractions[name][row][2]):{width}.4f}" for name in names)
            lines.append(line)
    lines += ["", "Loudest noise windows (ranking statistic)"]
    lines += [f"{name}  " + " ".join(f"{value:7.2f}" for value in np.sort(background[name])[::-1][:5]) for name in names]
    return "\n".join(lines)
