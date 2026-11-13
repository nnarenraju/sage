"""
``sage search-init``, ``search`` and ``statmap``: searches of observing
runs inside a run directory.
"""

import argparse
import os

import yaml

from sage.cli.registry import CommandError, closest, command
from sage.config import machine
from sage.config.loader import load_run_record
from sage.config.run_dir import SettingsChanged, find_run_dir
from sage.core.progress import print_steps


@command(
    "search-init",
    "Make a search of an observing run with one of the run's stored checkpoints",
)
def search_init(arguments: list) -> None:
    """
    Check the run, checkpoint and search data; write ``search.yaml``.

    Parameters
    ----------
    arguments : list of str
        Command-line arguments after ``sage search-init``.

    Raises
    ------
    CommandError
        If the run, checkpoint, machine file or search data is missing
        or invalid, or the search exists.
    """
    # Deferred: both import pycbc.
    from sage.data.download import DATASET
    from sage.search.settings import (
        check_search_data,
        initial_settings,
        search_dir,
        write_settings,
    )

    parser = argparse.ArgumentParser(
        prog="sage search-init",
        description="Make a search of an observing run with one of the run's "
        "stored checkpoints.",
    )
    parser.add_argument("observing_run", help="observing run to search, e.g. o3a")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="stored checkpoint, e.g. ema_calibrated, best or epoch_0127",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="run directory, or a directory inside it (default: here)",
    )
    options = parser.parse_args(arguments)
    run = options.observing_run.lower()
    again = f"sage search-init {run} --checkpoint {options.checkpoint}"

    directory = find_run_dir(options.directory)
    if not (directory / "run.yaml").exists():
        raise CommandError(
            f"no run.yaml in {directory} or above it",
            "cd <run directory>, then: " + again,
        )
    stored = sorted(path.stem for path in (directory / "checkpoints").glob("*.pt"))
    if options.checkpoint not in stored:
        match = closest(options.checkpoint, stored) or "<checkpoint>"
        raise CommandError(
            f"no checkpoint {options.checkpoint!r} in {directory / 'checkpoints'}; "
            f"stored: {', '.join(stored) or 'none'}",
            f"sage search-init {run} --checkpoint {match}",
        )

    try:
        data_dir = machine.read_data_dir()
    except (FileNotFoundError, ValueError) as error:
        raise CommandError(str(error), "sage setup --data-dir <directory>") from None
    record = (directory / "resolved" / "run.yaml").read_text()
    try:
        check_search_data(data_dir, run, load_run_record(record), DATASET)
    except machine.MissingData as error:
        raise CommandError(str(error), f"sage download {run} --search") from None
    except ValueError as error:
        release = machine.observing_run_dir(data_dir, run, "search")
        fix = f"mv {release} <elsewhere>, then: sage download {run} --search"
        raise CommandError(str(error), fix) from None

    search = search_dir(directory, f"{run}_{options.checkpoint}")
    if search.exists():
        raise CommandError(f"{search} exists", f"sage search {search.name}")
    path = write_settings(search, initial_settings(run, options.checkpoint))

    here = os.path.abspath(find_run_dir(".")) == os.path.abspath(directory)
    following = f"sage search {search.name}" + ("" if here else f" {directory}")
    print(f"Wrote {path}; edit it before the first sage search")
    print(f"Next: {following}")


@command("search", "Run or resume a search made by sage search-init")
def search(arguments: list) -> None:
    """
    Check the search and its data, then score zero-lag and background.

    Parameters
    ----------
    arguments : list of str
        Command-line arguments after ``sage search``.

    Raises
    ------
    CommandError
        If the search, machine file or data is missing or invalid, its
        GPU is not visible or ``search.yaml`` changed.
    """
    # Deferred: these import torch and pycbc.
    import torch

    from sage.data.download import DATASET
    from sage.search.manager import STEPS, run_search
    from sage.search.settings import check_search_data, read_settings, search_dir

    parser = argparse.ArgumentParser(
        prog="sage search",
        description="Run or resume a search made by sage search-init.",
    )
    parser.add_argument("name", help="search name, e.g. o3a_ema_calibrated")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="run directory, or a directory inside it (default: here)",
    )
    options = parser.parse_args(arguments)
    again = f"sage search {options.name}"

    directory = find_run_dir(options.directory)
    path = search_dir(directory, options.name)
    if not (path / "search.yaml").exists():
        made = sorted(found.parent.name for found in path.parent.glob("*/search.yaml"))
        raise CommandError(
            f"no search {options.name!r} in {path.parent}; "
            f"made: {', '.join(made) or 'none'}",
            "sage search-init <observing run> --checkpoint <checkpoint>",
        )
    try:
        data_dir = machine.read_data_dir()
    except (FileNotFoundError, ValueError) as error:
        raise CommandError(str(error), "sage setup --data-dir <directory>") from None
    try:
        settings = read_settings(path)
    except (ValueError, yaml.YAMLError) as error:
        fix = f"edit {path / 'search.yaml'} as above, then: {again}"
        raise CommandError(str(error), fix) from None

    run = settings["observing_run"]
    record = (directory / "resolved" / "run.yaml").read_text()
    try:
        check_search_data(data_dir, run, load_run_record(record), DATASET)
    except machine.MissingData as error:
        raise CommandError(str(error), f"sage download {run} --search") from None
    except ValueError as error:
        release = machine.observing_run_dir(data_dir, run, "search")
        fix = f"mv {release} <elsewhere>, then: sage download {run} --search"
        raise CommandError(str(error), fix) from None
    cuda = torch.device(settings["device"]).type == "cuda"
    if cuda and not torch.cuda.is_available():
        raise CommandError(
            f"search.yaml asks for {settings['device']}, and no GPU is visible here",
            f"put `{again}` in your job script, then: "
            "sage slurm-chain <job script> --jobs <n>",
        )

    checkpoint = f"checkpoints/{settings['checkpoint']}.pt"
    print(f"Searching {machine.gwosc_label(run)} with {checkpoint}, into {path}:")
    print_steps(STEPS, "Every step resumes where a stopped job left off.")
    try:
        scored = run_search(directory, options.name, data_dir)
    except SettingsChanged as error:
        raise CommandError(
            str(error),
            f"restore {path / 'search.yaml'}, or make another search with "
            "sage search-init",
        ) from None
    print()
    print(f"Zero-lag and background scored: {scored}")


@command("statmap", "False-alarm rates of a search's triggers, from its finished rows")
def statmap(arguments: list) -> None:
    """
    Rank the finished zero-lag triggers against the background.

    Parameters
    ----------
    arguments : list of str
        Command-line arguments after ``sage statmap``.

    Raises
    ------
    CommandError
        If the search has no scores, the machine file is invalid or
        ``run_statmap`` rejects the options.
    """
    # Deferred: imports torch and pycbc.
    from sage.search.statmap import CLUSTERING, REMOVE_NA, STEPS, run_statmap

    parser = argparse.ArgumentParser(
        prog="sage statmap",
        description="False-alarm rates of a search's triggers, from its finished rows.",
    )
    parser.add_argument("name", help="search name, e.g. o3a_ema_calibrated")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="run directory, or a directory inside it (default: here)",
    )
    parser.add_argument(
        "--clustering",
        choices=list(CLUSTERING),
        default="greedy",
        help="greedy: loudest first, its neighbours dropped; pycbc: kept if no "
        "louder trigger within the window (default: greedy)",
    )
    parser.add_argument(
        "--remove-na",
        choices=REMOVE_NA,
        default="off",
        help="drop triggers whose NA output is above their ranking statistic, "
        "before or after clustering (default: off)",
    )
    options = parser.parse_args(arguments)

    directory = find_run_dir(options.directory)
    path = directory / "search" / options.name
    if not (path / "zerolag" / "scores.json").exists():
        raise CommandError(
            f"no scores of a search {options.name!r} in {path.parent}",
            f"sage search {options.name}",
        )
    try:
        data_dir = machine.read_data_dir()
    except (FileNotFoundError, ValueError) as error:
        raise CommandError(str(error), "sage setup --data-dir <directory>") from None

    print(
        f"False-alarm rates of {options.name}, {options.clustering} clustering, "
        f"NA removal {options.remove_na}:"
    )
    print_steps(
        STEPS, "Only finished rows are read, so this runs while the search scores."
    )
    try:
        written = run_statmap(
            directory, options.name, data_dir, options.clustering, options.remove_na
        )
    except ValueError as error:
        fix = f"sage statmap {options.name} --remove-na off"
        raise CommandError(str(error), fix) from None
    print()
    print(f"Wrote {written}")

