"""
``sage download`` and ``sage build-asds``: data and ASDs in the data
directory.
"""

import argparse
import os
from pathlib import Path

from sage.cli.registry import CommandError, closest, command
from sage.cli.training import on
from sage.config import machine
from sage.config.run_dir import find_run_dir
from sage.core.progress import print_steps

WORKERS = 16


def directory_needs(directory: Path, data_dir: Path, name: str) -> tuple:
    """
    What a run directory reads, per observing run and for the fiducial.

    A run read only for the fiducial covers the detectors GWOSC lists
    for it.

    Parameters
    ----------
    directory : Path
        Run directory.
    data_dir : Path
        The machine's data directory.
    name : str
        Command asking, for the fix of an invalid run file.

    Returns
    -------
    tuple
        ``(needs, (fiducial runs, detectors))``. ``needs`` maps each
        observing run to ``(detectors, segments, bank)``: whether it
        reads the per-segment ASDs, and needs its recolour bank (as the
        recolour target or a run the fiducial pools).

    Raises
    ------
    CommandError
        If the run file is invalid.
    """
    from sage.data.download import run_detectors
    from sage.factory.build import asd_runs, bank_runs, strain_runs
    from sage.factory.train_run import check_run

    try:
        settings = check_run(directory)[0]
    except (ValueError, FileNotFoundError) as error:
        again = on(name, str(directory))
        fix = f"edit {directory / 'run.yaml'} as above, then: {again}"
        raise CommandError(str(error), fix) from None
    noise, detectors = settings["noise"], settings["data"]["detectors"]
    strain, asds, banks = strain_runs(noise), asd_runs(noise), bank_runs(noise)
    fiducial = sorted({run.lower() for run in noise["fiducial"]})
    needs = {}
    for run in sorted({*strain, *banks, *fiducial}):
        read = detectors if run in strain or run in banks else [d for d in detectors if d in run_detectors(data_dir, run)]
        needs[run] = (read, run in asds, run in banks or run in fiducial)
    return needs, (fiducial, detectors)


def run_needs(run: str, detectors: list | None, data_dir: Path) -> dict:
    """
    The given detectors of an observing run, or every one GWOSC lists.

    Parameters
    ----------
    run : str
        Observing run, in any case.
    detectors : list of str or None
        Detectors; None takes every detector GWOSC lists for the run.
    data_dir : Path
        The machine's data directory.

    Returns
    -------
    dict
        Lower-case run name -> detectors.

    Raises
    ------
    CommandError
        If the run is neither recorded nor listed by GWOSC.
    """
    from sage.data.download import gwosc_runs, recorded_gwosc, run_detectors

    if recorded_gwosc(data_dir, run.lower()) is None:
        names = gwosc_runs()
        if run.lower() not in names:
            match = closest(run.lower(), list(names))
            raise CommandError(f"observing run {run!r}: GWOSC lists {', '.join(sorted(names))}",
                               f"sage download {match or 'o3b'} --train")
    return {run.lower(): detectors or run_detectors(data_dir, run.lower())}


@command("download", "Download the data an observing run or a run directory needs", journey=3)
def download(arguments: list) -> None:
    """
    Download the data an observing run or a run directory needs.

    Each run's numbers are recorded beside its data.

    Parameters
    ----------
    arguments : list of str
        Command-line arguments after ``sage download``.

    Raises
    ------
    CommandError
        If the data directory is not set or writable, GWOSC cannot be
        reached or the request is invalid.
    """
    from requests.exceptions import RequestException

    parser = argparse.ArgumentParser(prog="sage download", description="Download the data an observing run or a run "
                                     "directory needs.")
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="observing run (e.g. o3b) with --train or --search, or a run "
        "directory or a directory inside it (default: here)",
    )
    data = parser.add_mutually_exclusive_group()
    data.add_argument(
        "--train",
        action="store_true",
        help="with an observing run: training strain (512 s pieces, events cut "
        "out), its bad time and zero spans",
    )
    data.add_argument(
        "--search",
        action="store_true",
        help="with an observing run: search strain (whole GWOSC segments, "
        "nothing removed), its bad time and zero spans",
    )
    parser.add_argument("--detectors", nargs="+", help="with an observing run: these detectors only (default: all GWOSC lists)")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"parallel downloads and fiducial line tests (default: {WORKERS})")
    options = parser.parse_args(arguments)
    try:
        data_dir = machine.read_data_dir()
    except (FileNotFoundError, ValueError) as error:
        raise CommandError(str(error), "sage setup --data-dir <directory>") from None
    if not os.access(data_dir, os.W_OK | os.X_OK):
        raise CommandError(f"the data directory {data_dir} is not writable here", f"chmod u+rwx {data_dir}")

    flag = " --search" if options.search else " --train" if options.train else ""
    again = f"sage download {options.target}{flag}"
    try:
        fetch(options, data_dir, again)
    except RequestException as error:
        raise CommandError(f"GWOSC could not be reached from here: {error}",
                           f"once https://gwosc.org is reachable: {again}") from None


def fetch(options: argparse.Namespace, data_dir: Path, again: str) -> None:
    """
    Download or adopt the strain ``options.target`` needs; print it.

    Parameters
    ----------
    options : argparse.Namespace
        Options of ``sage download``.
    data_dir : Path
        The machine's data directory.
    again : str
        The command as given, for the fix of a failed download.

    Raises
    ------
    CommandError
        If the target or options are invalid or data is missing.
    """
    from sage.data.download import TRAINING_STEPS, download_run

    # --train or --search names an observing run, even where a
    # directory has the same name; a run directory takes neither
    target = Path(options.target)
    if options.search:
        fetch_search(options, data_dir)
        return
    if options.train:
        needs = run_needs(options.target, options.detectors, data_dir)
        following = (f"sage build-asds <run directory>, or by observing run: "
                     f"sage build-asds {options.target.lower()} --segments")
    elif target.is_dir():
        target = find_run_dir(target)
        if not (target / "run.yaml").exists():
            raise CommandError(f"no run.yaml in {target}", "sage init <directory> --preset <name>")
        if options.detectors:
            raise CommandError("--detectors applies to an observing run; a run directory reads its own",
                               on("download", options.target))
        needs = directory_needs(target, data_dir, "download")[0]
        needs = {run: detectors for run, (detectors, _, _) in needs.items()}
        following = on("build-asds", str(target))
    else:
        raise CommandError(
            "name the data to download: --train or --search",
            f"sage download {options.target} --train",
        )

    reads = ", ".join(f"{run} ({' '.join(found)})" for run, found in needs.items())
    print(f"Downloading the training strain of {reads} into {data_dir}. For each run "
          "and detector:")
    print_steps(
        TRAINING_STEPS, "Then write each run's record. A stopped download resumes."
    )

    for run, detectors in needs.items():
        try:
            summaries = download_run(data_dir, run, detectors, options.workers)
        except machine.MissingData as error:
            raise CommandError(str(error), again) from None
        for detector, summary in summaries.items():
            first, last = summary["gps"]
            print(f"{run} {detector}: {summary['pieces']} pieces, {summary['days']:.2f} days, GPS {first:.0f}-{last:.0f}")
        print(f"Recorded in {machine.dataset_file(data_dir, run)}")
        print()
    print(f"Next: {following}")


def fetch_search(options: argparse.Namespace, data_dir: Path) -> None:
    """
    Download or resume an observing run's search strain; print it.

    Parameters
    ----------
    options : argparse.Namespace
        Options of ``sage download``.
    data_dir : Path
        The machine's data directory.

    Raises
    ------
    CommandError
        If GWOSC does not list the run or one of the detectors.
    """
    from sage.data.download import download_search, gwosc_detectors, gwosc_runs
    from sage.search.dataprep.release import DOWNLOAD_STEPS

    names = gwosc_runs()
    run = options.target.lower()
    if run not in names:
        match = closest(run, list(names)) or "o3a"
        raise CommandError(
            f"observing run {options.target!r}: GWOSC lists {', '.join(sorted(names))}",
            f"sage download {match} --search",
        )
    listed = gwosc_detectors(names[run])
    detectors = options.detectors or listed
    unknown = sorted(set(detectors) - set(listed))
    if unknown:
        raise CommandError(
            f"--detectors {' '.join(unknown)}: GWOSC lists {' '.join(listed)} "
            f"for {run}",
            f"sage download {run} --search --detectors {' '.join(listed)}",
        )

    release = machine.observing_run_dir(data_dir, run, "search")
    print(f"Downloading the {names[run]} search strain of {', '.join(detectors)} "
          f"into {release}. For each detector:")
    print_steps(DOWNLOAD_STEPS, "Then write the record. A stopped download resumes.")

    summaries = download_search(data_dir, run, detectors, options.workers)
    for detector, summary in summaries.items():
        first, last = summary["gps"]
        print(f"{run} {detector}: {summary['segments']} segments, "
              f"{summary['days']:.2f} days, GPS {first:.0f}-{last:.0f}")
    print(f"Recorded in {machine.search_dataset_file(data_dir, run)}; bad time "
          "and zero spans saved beside the strain")
    print()
    print(f"Next: cd <run directory> && sage search-init {run} "
          "--checkpoint <checkpoint>")


@command(
    "build-asds",
    "Build the per-segment ASDs, recolour bank or fiducial from downloaded strain",
    journey=4,
)
def build_asds(arguments: list) -> None:
    """
    Build or adopt ASDs for observing runs, or every one a run reads.

    Parameters
    ----------
    arguments : list of str
        Command-line arguments after ``sage build-asds``.

    Raises
    ------
    CommandError
        If the data directory is not set or writable, or the request
        is invalid.
    """
    parser = argparse.ArgumentParser(
        prog="sage build-asds",
        description="Build the per-segment ASDs, recolour bank or fiducial from "
        "downloaded strain.",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=["."],
        help="observing runs (e.g. o3b; the runs to pool with --fiducial) with "
        "--segments, --recolour-bank or --fiducial, or a run directory or a "
        "directory inside it (default: here)",
    )
    kind = parser.add_mutually_exclusive_group()
    kind.add_argument(
        "--segments", action="store_true", help="per-segment ASDs of each observing run"
    )
    kind.add_argument(
        "--recolour-bank",
        action="store_true",
        help="recolour bank of each observing run",
    )
    kind.add_argument(
        "--fiducial",
        action="store_true",
        help="fiducial ASDs pooled over the observing runs",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        help="with observing runs: these detectors only (default: all GWOSC lists)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help=f"processes for the fiducial's line tests (default: {WORKERS})",
    )
    options = parser.parse_args(arguments)
    try:
        data_dir = machine.read_data_dir()
    except (FileNotFoundError, ValueError) as error:
        raise CommandError(str(error), "sage setup --data-dir <directory>") from None
    if not os.access(data_dir, os.W_OK | os.X_OK):
        raise CommandError(
            f"the data directory {data_dir} is not writable here",
            f"chmod u+rwx {data_dir}",
        )

    # A kind names observing runs, even where a directory has the same
    # name; a run directory takes none
    first = Path(options.targets[0])
    kind = options.segments or options.recolour_bank or options.fiducial
    if not kind and len(options.targets) == 1 and first.is_dir():
        build_directory(options, find_run_dir(first), data_dir)
    else:
        build_runs(options, data_dir)


def build_directory(
    options: argparse.Namespace, directory: Path, data_dir: Path
) -> None:
    """
    Build or adopt every ASD the run directory reads; print each part.

    Parameters
    ----------
    options : argparse.Namespace
        Options of ``sage build-asds``.
    directory : Path
        Run directory.
    data_dir : Path
        The machine's data directory.

    Raises
    ------
    CommandError
        If there is no ``run.yaml``, ``--detectors`` is given or strain
        is missing.
    """
    from sage.data.download import build_asds as build, build_fiducials

    if not (directory / "run.yaml").exists():
        raise CommandError(
            f"no run.yaml in {directory}", "sage init <directory> --preset <name>"
        )
    if options.detectors:
        raise CommandError(
            "a run directory builds every ASD it reads; --detectors applies to "
            "observing runs",
            on("build-asds", str(directory)),
        )

    needs, fiducial = directory_needs(directory, data_dir, "build-asds")
    parts = []
    for run, (detectors, segments, bank) in needs.items():
        kinds = [kind for kind, wanted in (("per-segment ASDs", segments),
                                           ("recolour bank", bank)) if wanted]
        if kinds:
            parts.append(
                f"Build the {' and '.join(kinds)} of {run} ({' '.join(detectors)})"
            )
    fiducial_runs, fiducial_detectors = fiducial
    parts.append(f"Build the fiducial pooled over {', '.join(fiducial_runs)} "
                 f"({' '.join(fiducial_detectors)})")
    print(f"Building every ASD {directory} reads, into {data_dir / 'asds'}:")
    print_steps(parts, "A part already built is checked, not built again.")

    try:
        for run, (detectors, segments, bank) in needs.items():
            if segments or bank:
                record = build(data_dir, run, detectors, segments, bank)
                print(f"{run}: ASDs checked; recorded in {record}\n")
        fiducial_dir = build_fiducials(data_dir, *fiducial, options.workers)
    except machine.MissingData as error:
        raise CommandError(str(error), on("download", str(directory))) from None
    print(f"Fiducial ASDs checked: {fiducial_dir}")
    print()
    print(f"Next: {on('prepare', str(directory))}")


def build_runs(options: argparse.Namespace, data_dir: Path) -> None:
    """
    Build or adopt one kind of ASD for observing runs; print records.

    Parameters
    ----------
    options : argparse.Namespace
        Options of ``sage build-asds``.
    data_dir : Path
        The machine's data directory.

    Raises
    ------
    CommandError
        If no kind is named, or strain or ASDs are missing.
    """
    from sage.data.download import (
        BANK_STEPS,
        FIDUCIAL_STEPS,
        SEGMENT_STEPS,
        build_asds as build,
        build_fiducials,
        run_detectors,
    )

    runs = [target.lower() for target in options.targets]
    named = " ".join(runs)
    if not (options.segments or options.recolour_bank or options.fiducial):
        raise CommandError(
            "name the ASDs to build: --segments, --recolour-bank or --fiducial",
            f"sage build-asds {named} --segments",
        )
    for run in runs:
        if not machine.dataset_file(data_dir, run).exists():
            raise CommandError(
                f"no training strain of {run} in {data_dir}",
                f"sage download {run} --train",
            )

    if options.fiducial:
        listed = {detector for run in runs for detector in run_detectors(data_dir, run)}
        detectors = options.detectors or sorted(listed)
        print(f"Building the fiducial pooled over {', '.join(runs)} for "
              f"{' '.join(detectors)}, into {machine.fiducial_dir(data_dir, runs)}. "
              "For each detector:")
        print_steps(
            FIDUCIAL_STEPS, "A fiducial already built is checked, not built again."
        )
        try:
            fiducial = build_fiducials(data_dir, runs, detectors, options.workers)
        except machine.MissingASDs as error:
            fix = f"sage build-asds {named} --recolour-bank"
            raise CommandError(str(error), fix) from None
        except machine.MissingData as error:
            fix = " && ".join(f"sage download {run} --train" for run in runs)
            raise CommandError(str(error), fix) from None
        print(f"Fiducial ASDs checked: {fiducial}")
        print()
        print("Next: sage prepare <run directory>")
        return

    kind = "per-segment ASDs" if options.segments else "recolour bank"
    detectors = {run: options.detectors or run_detectors(data_dir, run) for run in runs}
    reads = ", ".join(f"{run} ({' '.join(found)})" for run, found in detectors.items())
    print(f"Building the {kind} of {reads}, into {data_dir / 'asds'}. For each run and "
          "detector:")
    print_steps(SEGMENT_STEPS if options.segments else BANK_STEPS,
                "Then write each run's record. Parts already built are checked, not "
                "built again.")
    for run in runs:
        try:
            record = build(
                data_dir, run, detectors[run], options.segments, options.recolour_bank
            )
        except machine.MissingData as error:
            raise CommandError(str(error), f"sage download {run} --train") from None
        print(f"{run}: {kind} checked; recorded in {record}\n")
    following = "--recolour-bank" if options.segments else "--fiducial"
    print(f"Next: sage build-asds {named} {following}")

