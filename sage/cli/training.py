"""``sage check``, ``sage train``, ``sage stop``, ``sage status`` and ``sage log``: training a run and following it."""

import argparse
import collections
import datetime
import json
import os
import re
import time
from pathlib import Path

import h5py
import yaml

from sage.cli.registry import CommandError, command
from sage.config import machine
from sage.config.run_dir import SettingsChanged, StateMismatch
from sage.utils.schedulers import schedulers

UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
CHECK_ONLY = "SAGE_CHECK_ONLY"  # 1 in a job: sage train runs the checks and one training step, then exits


def on(name: str, directory: str) -> str:
    """``sage <name>`` acting on ``directory``, which is left out when it is the current one."""
    return f"sage {name}" if Path(directory) == Path(".") else f"sage {name} {directory}"


def parse(name: str, summary: str, arguments: list, *options) -> argparse.Namespace:
    """Parse a run-directory command's arguments; each option is ``(flags, keywords)``."""
    parser = argparse.ArgumentParser(prog=f"sage {name}", description=f"{summary}.")
    parser.add_argument("directory", nargs="?", default=".", help="run directory (default: here)")
    for flags, keywords in options:
        parser.add_argument(*flags, **keywords)
    return parser.parse_args(arguments)


def seconds(duration: str) -> float:
    """A duration such as ``47h``, ``90m`` or ``2d`` in seconds."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", duration)
    if match is None:
        raise ValueError(f"--stop-after {duration}: give a number and one unit of {', '.join(UNITS)}, e.g. 47h")
    return float(match[1]) * UNITS[match[2]]


def checked_settings(name: str, directory: str) -> dict:
    """The run's settings, after the checks ``sage <name>`` makes first: run file, machine file, settings, data."""
    from sage.factory.train_run import check_data, check_run

    run_file = Path(directory) / "run.yaml"
    if not run_file.exists():
        raise CommandError(f"no run.yaml in {directory}", "sage init <directory> --preset <name>")
    try:
        data_dir = machine.read_data_dir()
    except (FileNotFoundError, ValueError) as error:
        raise CommandError(str(error), "sage machine-setup") from None
    if not os.access(data_dir, os.R_OK | os.X_OK):
        raise CommandError(f"the data directory {data_dir} is not readable here", f"chmod u+rx {data_dir}")
    try:
        settings = check_run(Path(directory))[0]
        check_data(Path(directory), settings, data_dir)
    except machine.MissingData as error:
        raise CommandError(str(error), on("download", directory)) from None
    except (ValueError, FileNotFoundError, yaml.YAMLError) as error:
        raise CommandError(str(error), f"edit {run_file} as above, then: {on(name, directory)}") from None
    return settings


def device_available(settings: dict, fix: str = "sage chain <your job script> --jobs 1") -> None:
    """Refuse a device or precision this machine lacks; the fix is by default a job on one that has it."""
    from sage.factory.build import check_device

    try:
        check_device(settings["train"])
    except ValueError as error:
        raise CommandError(str(error), fix) from None


def explained(name: str, directory: str, function, *args):
    """``function(*args)``, with a changed record turned into the command that fixes it."""
    try:
        return function(*args)
    except SettingsChanged as error:
        raise CommandError(str(error), f"sage clone {directory} <new directory> --full-state") from None
    except StateMismatch as error:
        raise CommandError(str(error), f"edit {Path(directory) / 'run.yaml'} back as above, then: {on(name, directory)}") from None


def checked_step(name: str, directory: str, in_job: bool) -> None:
    """
    Run every start-up check, then one real training step, and print the loss; write nothing.

    Outside a job a missing GPU moves the step to the CPU; in a job it is an error.
    """
    import torch

    from sage.core.logger import setup_logging
    from sage.factory.train_run import check_training

    settings = checked_settings(name, directory)
    cpu = not in_job and torch.device(settings["train"]["device"]).type == "cuda" and not torch.cuda.is_available()
    if cpu:
        print("No GPU is visible here: the training step runs on the CPU, uncompiled")
    elif in_job:
        device_available(settings, "request it in your job script, then: sage check --in-job <your job script>")
    else:
        device_available(settings)
    setup_logging(None)
    loss = explained(name, directory, check_training, Path(directory), cpu)
    device = "the CPU" if cpu else settings["train"]["device"]
    print(f"All checks passed; one training step on {device}: loss {loss:.5f}")


def submit_check(directory: str, script: str) -> None:
    """Run the checks that need no GPU here, then submit ``script`` once with ``SAGE_CHECK_ONLY=1``."""
    from sage.utils.slurm.chain import chain
    from sage.utils.slurm.query import job_output

    script = Path(script)
    if not script.is_file():
        raise CommandError(f"no job script {script}", f"{on('check', directory)} --in-job <your job script>")
    checked_settings("check", directory)
    found = schedulers()
    if found != ["slurm"]:
        raise CommandError("no scheduler to submit to (no sbatch on PATH)" if not found
                           else f"submitting on {', '.join(found)} is not available yet", on("check", directory))
    try:
        job = chain(script, 1, environment={CHECK_ONLY: "1"})[0]
    except RuntimeError as error:
        raise CommandError(str(error), f"{on('check', directory)} --in-job {script}") from None
    output = job_output(job)
    print(f"Submitted check job {job}: {script} runs with {CHECK_ONLY}=1, so its sage train runs every check and one "
          "training step, writes nothing and exits")
    if output:
        print(f"Its output: {output}")
    print(f"Next, once it has passed: sage chain {script} --jobs <number of jobs>")


@command("check", "Run every start-up check, then one real training step; writes nothing", journey=4)
def check(arguments: list) -> None:
    """Check the run as ``sage train`` would start it and take one training step, here or in a job."""
    options = parse(
        "check", "Run every start-up check, then one real training step; writes nothing", arguments,
        (["--in-job"], {"metavar": "SCRIPT", "help": "submit your job script once in check-only mode, so the "
                        "checks and the step run on a compute node"}),
    )
    if options.in_job is not None:
        submit_check(options.directory, options.in_job)
        return
    checked_step("check", options.directory, in_job=False)
    print(f"Next: {on('train', options.directory)}")


@command("train", "Train the run, resuming from its latest checkpoint", journey=5)
def train(arguments: list) -> None:
    """Train the run to its last epoch, then run the final check; ``--stop-after`` ends a job segment early."""
    from sage.factory.train_run import STOP_FILE, train_run
    from sage.utils.checkpoint import next_epoch

    options = parse(
        "train", "Train the run, resuming from its latest checkpoint", arguments,
        (["--stop-after"], {"help": "save and exit before an epoch that would not end in time, e.g. 47h"}),
    )
    check_only = os.environ.get(CHECK_ONLY)
    if check_only is not None:
        if check_only != "1":
            raise CommandError(f"{CHECK_ONLY}={check_only}: only 1 is accepted (check only)",
                               f"unset {CHECK_ONLY} && {on('train', options.directory)}")
        print(f"{CHECK_ONLY}=1: checks and one training step only; nothing is trained or written", flush=True)
        checked_step("train", options.directory, in_job=True)
        print("Next: sage chain <your job script> --jobs <number of jobs>")
        return
    try:
        stop_after = None if options.stop_after is None else seconds(options.stop_after)
    except ValueError as error:
        raise CommandError(str(error), f"{on('train', options.directory)} --stop-after 47h") from None
    directory = Path(options.directory)
    stop_file = directory / STOP_FILE
    if stop_file.exists():
        raise CommandError(f"{directory} was stopped by sage stop", f"rm {stop_file} && {on('train', options.directory)}")
    settings = checked_settings("train", options.directory)
    device_available(settings)
    explained("train", options.directory, train_run, directory, stop_after)
    if (directory / "results" / "final_check.json").exists():
        print(f"Trained to the end. Next: {on('status', options.directory)}")
    elif stop_file.exists():
        print(f"Stopped by sage stop. Next, to train on: rm {stop_file} && {on('train', options.directory)}")
    elif next_epoch(directory) == settings["train"]["epochs"]:
        print(f"Stopped before the final check, which would not end in time; the next job runs it. "
              f"Next: {on('train', options.directory)}")
    else:
        print(f"Stopped before an epoch that would not end in time; the next job resumes. "
              f"Next: {on('train', options.directory)}")


@command("stop", "Stop the run's training after its current epoch")
def stop(arguments: list) -> None:
    """Write the stop file: training saves after its current epoch and exits, and no later job trains on."""
    from sage.factory.train_run import STOP_FILE

    options = parse("stop", "Stop the run's training after its current epoch", arguments)
    directory = Path(options.directory)
    if not (directory / "run.yaml").exists():
        raise CommandError(f"no run.yaml in {directory}", "sage init <directory> --preset <name>")
    (directory / STOP_FILE).touch()
    print(f"Wrote {directory / STOP_FILE}: training stops once its current epoch is saved, and later jobs do "
          f"not start. To train on: rm {directory / STOP_FILE} && {on('train', options.directory)}")
    print(f"Next: {on('status', options.directory)}")


@command("status", "Show how far the run has trained and its latest losses")
def status(arguments: list) -> None:
    """Print the epochs trained, the latest and lowest losses, the final check and when the log was last written."""
    from sage.config.loader import load_run_record
    from sage.factory.build import validates
    from sage.factory.train_run import FINAL_CHECK_ESTIMATE, STOP_FILE
    from sage.utils.checkpoint import next_epoch

    options = parse("status", "Show how far the run has trained and its latest losses", arguments)
    directory = Path(options.directory)
    done = next_epoch(directory)
    clone_file = directory / "resolved" / "clone.yaml"
    cloned = []
    if clone_file.exists():
        clone = yaml.safe_load(clone_file.read_text())
        state = "full state" if clone["state"] == "full" else "weights only"
        cloned = [f"Cloned from {clone['source']}, epoch {clone['epoch']} ({state})"]
    if done == 0:
        if not (directory / "run.yaml").exists():
            raise CommandError(f"no run.yaml in {directory}", "sage init <directory> --preset <name>")
        print("\n".join([f"{directory}: training has not started", *cloned]))
        print(f"Next: {on('train', options.directory)}")
        return

    settings = load_run_record((directory / "resolved" / "run.yaml").read_text())
    epochs = settings["train"]["epochs"]
    lines = [f"{directory}: {done} of {epochs} epochs trained", *cloned]
    try:
        with h5py.File(directory / "results" / "losses.h5", "r", locking=False) as file:
            training, validation = file["training"]["loss"][:done, 0], file["validation"]["loss"][:done, 0]
    except OSError:
        lines.append("Losses: the file is being written; run this again")
    else:
        lines.append(f"Training loss {training[-1]:.5f} (epoch {done - 1})")
        validated = [epoch for epoch in range(done) if validates(epoch, settings["validation"], epochs)]
        if validated:
            best = min(validated, key=lambda epoch: validation[epoch])
            lines.append(f"Validation loss {validation[validated[-1]]:.5f} (epoch {validated[-1]}); "
                         f"lowest {validation[best]:.5f} (epoch {best}, best.pt)")
    final = directory / "results" / "final_check.json"
    if final.exists():
        result = json.loads(final.read_text())
        for file in ("best.pt", "ema_calibrated.pt"):
            if file in result:
                lines.append(f"Final check, {file} (epoch {result[file]['epoch']}): loss "
                             f"{result[file]['loss']['total']:.5f} on {result['samples']} samples")
    elif (directory / "results" / FINAL_CHECK_ESTIMATE).exists():
        estimate = json.loads((directory / "results" / FINAL_CHECK_ESTIMATE).read_text())
        when = "not finished" if done == epochs else "runs after the last epoch"
        lines.append(f"Final check: {when}; takes about {datetime.timedelta(seconds=round(estimate['seconds']))} "
                     f"(estimated at epoch {estimate['epoch']}), so leave a job for it")
    elif done == epochs:
        lines.append("Final check: not finished")
    log = directory / "logs" / "run.log"
    if log.exists():
        written = datetime.datetime.fromtimestamp(log.stat().st_mtime).astimezone()
        age = datetime.timedelta(seconds=int(time.time() - log.stat().st_mtime))
        lines.append(f"Log last written {written:%Y-%m-%d %H:%M:%S %Z}, {age} ago")
    if (directory / STOP_FILE).exists():
        lines.append(f"Stopped by sage stop; to train on: rm {directory / STOP_FILE} && {on('train', options.directory)}")
    print("\n".join(lines))
    print(f"Next: {on('plot' if final.exists() else 'log', options.directory)}")


@command("log", "Print the end of the run's log, or follow it")
def log(arguments: list) -> None:
    """Print the last lines of ``logs/run.log``; with ``--follow``, keep printing new lines until Ctrl-C."""
    options = parse(
        "log", "Print the end of the run's log, or follow it", arguments,
        (["--lines"], {"type": int, "default": 20, "help": "lines to print (default: 20)"}),
        (["--follow"], {"action": "store_true", "help": "keep printing new lines until Ctrl-C"}),
    )
    path = Path(options.directory) / "logs" / "run.log"
    if not path.exists():
        raise CommandError(f"no log yet: {path}", on("train", options.directory))
    if options.lines < 1:
        raise CommandError(f"--lines {options.lines}: print at least one line", f"{on('log', options.directory)} --lines 20")
    with path.open() as file:
        print("".join(collections.deque(file, maxlen=options.lines)), end="")
        if not options.follow:
            print(f"Next: {on('status', options.directory)}")
            return
        try:
            while True:
                line = file.readline()
                if line:
                    print(line, end="", flush=True)
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            print()
