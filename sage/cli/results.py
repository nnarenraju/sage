"""
``sage validate``, ``benchmark``, ``plot`` and ``compare``: looking at
trained runs.
"""

import argparse
import shlex
from pathlib import Path

from sage.cli.registry import CommandError, command
from sage.cli.training import checked_settings, device_available, on, parse
from sage.config.run_dir import SettingsChanged, find_run_dir


@command("validate", "Validate one stored checkpoint on the run's final validation set")
def validate(arguments: list) -> None:
    """
    Validate one stored checkpoint on the final validation set.

    The checkpoint is ``best.pt``, a stored epoch or
    ``ema_calibrated.pt``; the pass is added to ``results/validate.h5``.

    Parameters
    ----------
    arguments : list of str
        Command-line arguments after ``sage validate``.

    Raises
    ------
    CommandError
        If the checkpoint is unavailable or a check fails, with the
        command that fixes it.
    """
    from sage.config.run_dir import training_started
    from sage.core.logger import setup_logging
    from sage.factory.clone import EpochUnavailable, stored_checkpoints
    from sage.factory.train_run import validate_checkpoint

    options = parse(
        "validate", "Validate one stored checkpoint on the run's final validation set", arguments,
        (["--from-epoch"], {"type": int, "help": "a stored epoch instead of best.pt"}),
        (["--ema-calibrated"], {"action": "store_true", "help": "ema_calibrated.pt instead of best.pt"}),
    )
    directory = Path(options.directory)
    if options.from_epoch is not None and options.ema_calibrated:
        raise CommandError("--from-epoch and --ema-calibrated each name a checkpoint; give one",
                           f"{on('validate', options.directory)} --ema-calibrated")
    settings = checked_settings("validate", options.directory)
    if not training_started(directory):
        raise CommandError(f"{directory}: no checkpoint to validate yet", on("train", options.directory))
    if options.ema_calibrated and not (directory / "checkpoints" / "ema_calibrated.pt").exists():
        if not settings["ema"]["enabled"]:
            raise CommandError(f"{directory}: EMA is not enabled, so there is no ema_calibrated.pt",
                               on("validate", options.directory))
        raise CommandError(f"{directory}: no ema_calibrated.pt yet; the final check writes it after the last epoch",
                           on("train", options.directory))
    device_available(settings, f"put `{on('validate', options.directory)}` in a job script on a node that has it, "
                               "then: sbatch <your job script>")
    setup_logging(None)
    again = on("validate", options.directory)
    try:
        result = validate_checkpoint(directory, options.from_epoch, options.ema_calibrated)
    except SettingsChanged as error:
        raise CommandError(str(error), f"edit {directory / 'run.yaml'} back as above, then: {again}") from None
    except EpochUnavailable as error:
        latest = max(stored_checkpoints(directory))
        raise CommandError(str(error), f"{again} --from-epoch {latest}") from None
    losses = ", ".join(f"{name} {value:.5f}" for name, value in result["loss"].items())
    print(f"Validated {result['file']} (epoch {result['epoch']}) on {result['samples']} validation samples: {losses}")
    print(f"Added to {directory / 'results' / 'validate.h5'} as group {result['group']}")
    print(f"Next: {on('plot', options.directory)}")


@command("benchmark", "Benchmark one stored checkpoint on MLGWSC-1 dataset 4 against every team")
def benchmark(arguments: list) -> None:
    """
    Benchmark one stored checkpoint on MLGWSC-1 dataset 4.

    Searches the dataset with the checkpoint and evaluates it as
    MLGWSC-1 does, against every team.

    Parameters
    ----------
    arguments : list of str
        Command-line arguments after ``sage benchmark``.

    Raises
    ------
    CommandError
        If no benchmark is named, the checkpoint is unavailable or a
        check fails, with the command that fixes it.
    """
    from sage.benchmark.mlgwsc1.mlgwsc1 import benchmark as mlgwsc1
    from sage.cli.data import WORKERS
    from sage.config.run_dir import training_started
    from sage.core.logger import setup_logging
    from sage.factory.clone import EpochUnavailable, stored_checkpoints

    summary = "Benchmark one stored checkpoint on MLGWSC-1 dataset 4 against every team"
    options = parse(
        "benchmark", summary, arguments,
        (["--mlgwsc1"], {"action": "store_true", "help": "the MLGWSC-1 dataset 4 benchmark"}),
        (["--from-epoch"], {"type": int, "help": "a stored epoch instead of best.pt"}),
        (["--ema-calibrated"], {"action": "store_true", "help": "ema_calibrated.pt instead of best.pt"}),
        (["--workers"], {"type": int, "default": WORKERS, "help": f"processes for the injection SNRs (default: {WORKERS})"}),
    )
    directory = Path(options.directory)
    again = f"{on('benchmark', options.directory)} --mlgwsc1"
    if not options.mlgwsc1:
        raise CommandError("name the benchmark to run; the one there is: --mlgwsc1", again)
    if options.from_epoch is not None and options.ema_calibrated:
        raise CommandError("--from-epoch and --ema-calibrated each name a checkpoint; give one", f"{again} --ema-calibrated")
    settings = checked_settings("benchmark", options.directory)
    if not training_started(directory):
        raise CommandError(f"{directory}: no checkpoint to benchmark yet", on("train", options.directory))
    device_available(settings, f"put `{again}` in a job script on a node that has it, then: sbatch <your job script>")
    setup_logging(None)
    try:
        output = mlgwsc1(directory, options.from_epoch, options.ema_calibrated, options.workers)
    except SettingsChanged as error:
        raise CommandError(str(error), f"edit {directory / 'run.yaml'} back as above, then: {again}") from None
    except EpochUnavailable as error:
        raise CommandError(str(error), f"{again} --from-epoch {max(stored_checkpoints(directory))}") from None
    except FileExistsError as error:
        raise CommandError(str(error), f"{again} --from-epoch <another stored epoch>") from None
    print(f"Benchmarked on MLGWSC-1 dataset 4: events and evaluation in {output}")
    print(f"Next: {on('status', options.directory)}")


@command("plot", "Draw the run's validation figures into results/plots", journey=8)
def plot(arguments: list) -> None:
    """
    Draw the figures of the best validated epoch, or another group.

    Parameters
    ----------
    arguments : list of str
        Command-line arguments after ``sage plot``.

    Raises
    ------
    CommandError
        If the run has no ``run.yaml`` or validation results, the group
        is unknown or a results file cannot be read.
    """
    import matplotlib

    from sage.plotting.manager import ValidationPlotManager

    options = parse(
        "plot", "Draw the run's validation figures into results/plots", arguments,
        (["--group"], {"help": "a validation group instead of the best validated epoch, e.g. epoch_0010, "
                       "final_best or validate_best"}),
    )
    directory = Path(options.directory)
    results = directory / "results"
    if not (directory / "run.yaml").exists():
        raise CommandError(f"no run.yaml in {directory}", "sage init <directory> --preset <name>")
    if not (results / "validation.h5").exists():
        raise CommandError(f"{directory}: no validation results yet", on("train", options.directory))
    files = [path for path in (results / "validation.h5", results / "validate.h5") if path.exists()]
    matplotlib.use("Agg")
    try:
        manager = ValidationPlotManager(files, results / "losses.h5", export_dir=str(results / "plots"))
        groups = manager.resolve_epochs(options.group)
        manager.make_all_plots(epochs=options.group)
    except KeyError as error:
        raise CommandError(error.args[0], f"{on('plot', options.directory)} --group <one of the groups above>") from None
    except OSError as error:
        raise CommandError(f"{error}; a results file may be being written", on("plot", options.directory)) from None
    print(f"Drew the figures for {', '.join(groups)} in {results / 'plots'}")
    print("Next: sage help")


@command("compare", "Score several runs' best.pt on identical validation windows")
def compare(arguments: list) -> None:
    """
    Score runs on the first run's windows; print and save the report.

    Parameters
    ----------
    arguments : list of str
        Command-line arguments after ``sage compare``.

    Raises
    ------
    CommandError
        If a run is missing, not started, named like another or does
        not share the first run's windows, or the output exists.
    """
    from sage.config.run_dir import training_started
    from sage.core.logger import setup_logging
    from sage.factory.compare import INCOMPLETE_FILE, OUTPUT_FILE, REPORT_FILE, WindowMismatch, compare_runs

    parser = argparse.ArgumentParser(prog="sage compare", description="Score several runs' best.pt on identical "
                                     "validation windows.")
    parser.add_argument(
        "runs",
        nargs="+",
        help="run directories, or directories inside them; the first "
        "one's validation settings define the windows",
    )
    parser.add_argument("--output", required=True, help=f"new directory for {OUTPUT_FILE}, {REPORT_FILE} and the figures")
    parser.add_argument("--noise-windows", type=int, default=10_000_000, help="noise-only windows (default: 10000000)")
    parser.add_argument("--signal-windows", type=int, default=1_000_000, help="windows with a signal (default: 1000000)")
    parser.add_argument("--seed", type=int, default=170817, help="seed of the windows (default: 170817)")
    options = parser.parse_args(arguments)
    again = shlex.join(["sage", "compare", *arguments])
    runs = [find_run_dir(run) for run in options.runs]
    for run in runs:
        if not (run / "run.yaml").exists():
            raise CommandError(f"no run.yaml in {run}", "sage init <directory> --preset <name>")
        if not training_started(run):
            raise CommandError(f"{run}: no checkpoint to compare yet", on("train", str(run)))
    names = [run.resolve().name for run in runs]
    if len(set(names)) < len(names):
        raise CommandError(f"runs are named by their directories, and these repeat: {', '.join(names)}",
                           "ln -s <run directory> <distinct name>, then compare the links")
    if options.noise_windows < 1 or options.signal_windows < 1:
        raise CommandError("give at least one noise and one signal window",
                           f"sage compare {shlex.join(options.runs)} --output {options.output}")
    output = Path(options.output)
    if output.exists():
        stopped = f"; {INCOMPLETE_FILE} there: that comparison stopped early" if (output / INCOMPLETE_FILE).exists() else ""
        raise CommandError(f"{output} exists{stopped}; every comparison writes into a new directory",
                           f"sage compare {shlex.join(options.runs)} --output <new directory>")
    settings = checked_settings("check", options.runs[0])
    device_available(settings, f"put `{again}` in a job script on a node that has it, then: sbatch <your job script>")
    setup_logging(None)
    try:
        text = compare_runs(runs, output, options.noise_windows, options.signal_windows, options.seed)
    except SettingsChanged as error:
        raise CommandError(str(error), f"edit that run.yaml back as above, then: {again}") from None
    except WindowMismatch as error:
        raise CommandError(str(error), f"sage compare <runs that share detectors and window> --output {options.output}") from None
    print(text)
    print(f"Wrote {OUTPUT_FILE}, {REPORT_FILE} and the figures in {output}")
    print("Next: sage help")
