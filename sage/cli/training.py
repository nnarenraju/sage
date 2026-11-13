"""``sage check``, ``sage train``, ``sage stop``, ``sage status`` and ``sage log``: training a run and following it."""

import argparse
import collections
import datetime
import json
import math
import os
import re
import time
import zoneinfo
from pathlib import Path

import h5py
import yaml

from sage.cli.registry import CommandError, command
from sage.config import machine
from sage.config.run_dir import NotPrepared, SettingsChanged, StateMismatch, find_run_dir, training_started

UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
TIMEZONES = {  # sage status --timezone: name -> IANA time zone
    "utc": "UTC",
    "uk": "Europe/London",
    "europe-central": "Europe/Berlin",
    "europe-eastern": "Europe/Athens",
    "us-eastern": "America/New_York",
    "us-central": "America/Chicago",
    "us-mountain": "America/Denver",
    "us-pacific": "America/Los_Angeles",
    "india": "Asia/Kolkata",
    "china": "Asia/Shanghai",
    "japan": "Asia/Tokyo",
    "australia-eastern": "Australia/Sydney",
    "australia-western": "Australia/Perth",
}
EPOCH_START = re.compile(r"^(\S+ \S+) \| .* \| Epoch \d+ \| training:")
JOB_START = re.compile(r" \| Run .* from epoch \d+, seed ")


def on(name: str, directory: str) -> str:
    """``sage <name>`` on ``directory``, omitted if found from here."""
    here = os.path.abspath(find_run_dir("."))
    if os.path.abspath(directory) == here:
        return f"sage {name}"
    return f"sage {name} {directory}"


def parse(name: str, summary: str, arguments: list, *options) -> argparse.Namespace:
    """Parse a run-directory command's arguments; each option is ``(flags, keywords)``."""
    parser = argparse.ArgumentParser(prog=f"sage {name}", description=f"{summary}.")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="run directory, or a directory inside it (default: here)",
    )
    for flags, keywords in options:
        parser.add_argument(*flags, **keywords)
    options = parser.parse_args(arguments)
    options.directory = str(find_run_dir(options.directory))
    return options


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
        raise CommandError(str(error), "sage setup --data-dir <directory>") from None
    if not os.access(data_dir, os.R_OK | os.X_OK):
        raise CommandError(f"the data directory {data_dir} is not readable here", f"chmod u+rx {data_dir}")
    try:
        settings = check_run(Path(directory))[0]
        check_data(Path(directory), settings, data_dir)
    except machine.MissingASDs as error:
        raise CommandError(str(error), on("build-asds", directory)) from None
    except machine.MissingData as error:
        raise CommandError(str(error), on("download", directory)) from None
    except (ValueError, FileNotFoundError, yaml.YAMLError) as error:
        raise CommandError(str(error), f"edit {run_file} as above, then: {on(name, directory)}") from None
    return settings


def device_available(settings: dict, fix: str = "run it on a machine that has it, e.g. from your job script") -> None:
    """Refuse a device or precision this machine lacks."""
    from sage.factory.build import check_device

    try:
        check_device(settings["train"])
    except ValueError as error:
        raise CommandError(str(error), fix) from None


def explained(name: str, directory: str, function, *args):
    """``function(*args)``, with a changed record turned into the command that fixes it."""
    from sage.factory.train_run import AlreadyRunning

    try:
        return function(*args)
    except AlreadyRunning as error:
        raise CommandError(str(error), on("status", directory)) from None
    except NotPrepared as error:
        raise CommandError(str(error), on("prepare", directory)) from None
    except SettingsChanged as error:
        raise CommandError(str(error), f"sage clone {directory} <new directory> --full-state") from None
    except StateMismatch as error:
        raise CommandError(str(error), f"edit {Path(directory) / 'run.yaml'} back as above, then: {on(name, directory)}") from None


def checked_step(name: str, directory: str) -> None:
    """
    Run every start-up check and one real training step, and print the loss; write only ``flags/checked``.

    The check is small; without a visible GPU it runs on the CPU.
    """
    import torch

    from sage.core.logger import setup_logging
    from sage.factory.build import compile_arguments
    from sage.factory.train_run import CHECK_STEPS, check_settings, check_training

    settings = checked_settings(name, directory)
    cpu = torch.device(settings["train"]["device"]).type == "cuda" and not torch.cuda.is_available()
    if not cpu:
        device_available(settings)
    small = check_settings(settings, cpu)
    where = "the CPU (no GPU is visible)" if cpu else small["train"]["device"]
    arguments = compile_arguments(small["train"]["compile"])
    compiled = "uncompiled" if arguments is None else f"compiled ({arguments['mode']}, minutes on the first step)"
    print(f"Checking on {where}, {compiled}, with a batch of {small['train']['batch_size']} and "
          f"{small['noise']['read_workers']} read workers")
    print("Steps:\n" + "\n".join(f"  {index}. {step}" for index, step in enumerate(CHECK_STEPS, 1)), flush=True)
    started = []

    def progress(index: int) -> None:
        """Print the time the previous step took, then the step starting."""
        if started:
            print(f"   done in {time.monotonic() - started[-1]:.1f} s", flush=True)
        print(f"[{index + 1}/{len(CHECK_STEPS)}] {CHECK_STEPS[index]}", flush=True)
        started.append(time.monotonic())

    setup_logging(None)
    loss = explained(name, directory, check_training, Path(directory), cpu, progress)
    print(f"   done in {time.monotonic() - started[-1]:.1f} s")
    device = "the CPU" if cpu else settings["train"]["device"]
    print(f"All checks passed; one training step on {device}: loss {loss:.5f}. Wrote flags/checked")


@command("prepare", "Check the run and write every setting it uses to resolved/", journey=5)
def prepare(arguments: list) -> None:
    """Write ``resolved/``: settings with every default, prior, fiducial copy and derived values; rerun after edits."""
    from sage.factory.train_run import prepare_run

    options = parse("prepare", "Check the run and write every setting it uses to resolved/", arguments)
    directory = Path(options.directory)
    checked_settings("prepare", options.directory)
    if training_started(directory):
        raise CommandError(f"{directory}: training has started, so its record in resolved/ is fixed",
                           f"sage clone {options.directory} <new directory> --full-state")
    try:
        prepare_run(directory)
    except machine.MissingASDs as error:
        raise CommandError(str(error), on("build-asds", options.directory)) from None
    except machine.MissingData as error:
        raise CommandError(str(error), on("download", options.directory)) from None
    except StateMismatch as error:
        raise CommandError(str(error), f"edit {directory / 'run.yaml'} back as above, then: "
                           f"{on('prepare', options.directory)}") from None
    print(f"Wrote {directory / 'resolved'}: run.yaml, prior.yaml, derived.yaml and fiducial/")
    print(f"Next: {on('check', options.directory)}")


@command("check", "Run every start-up check and one real training step; marks the run checked", journey=6)
def check(arguments: list) -> None:
    """Check the run as ``sage train`` would start it and take one training step: on a GPU if one is visible."""
    options = parse("check", "Run every start-up check and one real training step; marks the run checked", arguments)
    checked_step("check", options.directory)
    print(f"Next: {on('train', options.directory)}")


@command("train", "Train the run, resuming from its latest checkpoint", journey=7)
def train(arguments: list) -> None:
    """Train the run to its last epoch, then run the final check; ``--stop-after`` ends a job segment early."""
    from sage.factory.train_run import STOP_FLAG, flag, train_run
    from sage.utils.checkpoint import next_epoch

    options = parse(
        "train", "Train the run, resuming from its latest checkpoint", arguments,
        (["--stop-after"], {"help": "save and exit before an epoch that would not end in time, e.g. 47h"}),
    )
    try:
        stop_after = None if options.stop_after is None else seconds(options.stop_after)
    except ValueError as error:
        raise CommandError(str(error), f"{on('train', options.directory)} --stop-after 47h") from None
    directory = Path(options.directory)
    stop_file = flag(directory, STOP_FLAG)
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
    """Write ``flags/stop``: training saves after its current epoch and exits, and no later job trains on."""
    from sage.factory.train_run import STOP_FLAG, flag

    options = parse("stop", "Stop the run's training after its current epoch", arguments)
    directory = Path(options.directory)
    if not (directory / "run.yaml").exists():
        raise CommandError(f"no run.yaml in {directory}", "sage init <directory> --preset <name>")
    stop_file = flag(directory, STOP_FLAG)
    stop_file.parent.mkdir(exist_ok=True)
    stop_file.touch()
    print(f"Wrote {stop_file}: training stops once its current epoch is saved, and later jobs do "
          f"not start. To train on: rm {stop_file} && {on('train', options.directory)}")
    print(f"Next: {on('status', options.directory)}")


def flag_lines(directory: str, zone: zoneinfo.ZoneInfo) -> list:
    """The run's flags with when each was set (in ``zone``), and how to train on after ``sage stop``."""
    from sage.factory.train_run import FLAGS_DIR, STOP_FLAG, flag

    lines = []
    flags = sorted((Path(directory) / FLAGS_DIR).glob("*"))
    if flags:
        set_at = lambda path: datetime.datetime.fromtimestamp(path.stat().st_mtime, zone)
        lines.append("Flags: " + ", ".join(f"{path.name} (since {set_at(path):%Y-%m-%d %H:%M:%S %Z})" for path in flags))
    if flag(directory, STOP_FLAG).exists():
        lines.append(f"Stopped by sage stop; to train on: rm {flag(directory, STOP_FLAG)} && {on('train', directory)}")
    return lines


def epoch_seconds(log: Path, window: int) -> float | None:
    """
    Mean time per epoch over the last ``window`` epochs in ``run.log``.

    An epoch that spans two jobs is left out, so waits between jobs do
    not count.

    Parameters
    ----------
    log : Path
        The run's ``logs/run.log``.
    window : int
        Epochs to average over.

    Returns
    -------
    float or None
        Seconds per epoch; None before an epoch has finished.
    """
    durations, previous = [], None
    with open(log) as file:
        for line in file:
            if JOB_START.search(line):
                previous = None
            elif match := EPOCH_START.match(line):
                started = datetime.datetime.fromisoformat(match[1])
                if previous is not None:
                    durations.append((started - previous).total_seconds())
                previous = started

    recent = durations[-window:]
    return sum(recent) / len(recent) if recent else None


def finish_time(directory: Path, settings: dict, done: int) -> datetime.datetime | None:
    """
    When a running run finishes: its epochs left, then the final check.

    Epochs take their mean over one validation and mining cycle
    (``epoch_seconds``), from the later of the last epoch's end and the
    job's start.

    Parameters
    ----------
    directory : Path
        Run directory with ``flags/running``.
    settings : dict
        The run's resolved settings.
    done : int
        Epochs trained.

    Returns
    -------
    datetime.datetime or None
        Finish time in UTC; None before the first epoch has finished.
    """
    # Deferred: train_run imports torch.
    from sage.factory.train_run import FINAL_CHECK_ESTIMATE, RUNNING_FLAG, flag

    cycle = [settings["validation"]["every_n_epochs"]]
    if settings["mining"]["enabled"]:
        cycle.append(settings["mining"]["every_n_epochs"])
    mean = epoch_seconds(directory / "logs" / "run.log", math.lcm(*cycle))
    estimate = directory / "results" / FINAL_CHECK_ESTIMATE
    if mean is None or not estimate.exists():
        return None

    started = max(
        (directory / "checkpoints" / "progress.json").stat().st_mtime,
        flag(directory, RUNNING_FLAG).stat().st_mtime,
    )
    seconds = (settings["train"]["epochs"] - done) * mean
    seconds += json.loads(estimate.read_text())["seconds"]
    return datetime.datetime.fromtimestamp(started + seconds, datetime.timezone.utc)


@command("status", "Show how far the run has trained and its latest losses")
def status(arguments: list) -> None:
    """Print the epochs trained, the latest and lowest losses, the final check and when the log was last written."""
    from sage.config.loader import load_run_record
    from sage.factory.build import validates
    from sage.factory.train_run import FINAL_CHECK_ESTIMATE, RUNNING_FLAG, flag
    from sage.utils.checkpoint import next_epoch

    options = parse(
        "status", "Show how far the run has trained and its latest losses", arguments,
        (["--timezone"], {"choices": list(TIMEZONES), "default": "utc", "help": "time zone of the times shown (default: utc)"}),
    )
    zone = zoneinfo.ZoneInfo(TIMEZONES[options.timezone])
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
        print("\n".join([f"{directory}: training has not started", *cloned, *flag_lines(options.directory, zone)]))
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
    if flag(directory, RUNNING_FLAG).exists() and not final.exists():
        finish = finish_time(directory, settings, done)
        lines.append("Finish: estimated once an epoch has finished" if finish is None else
                     f"Finishes about {finish.astimezone(zone):%Y-%m-%d %H:%M %Z}, final check included; "
                     "waits between jobs not counted")
    log = directory / "logs" / "run.log"
    if log.exists():
        written = datetime.datetime.fromtimestamp(log.stat().st_mtime, zone)
        age = datetime.timedelta(seconds=int(time.time() - log.stat().st_mtime))
        lines.append(f"Log last written {written:%Y-%m-%d %H:%M:%S %Z}, {age} ago")
    lines += flag_lines(options.directory, zone)
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
