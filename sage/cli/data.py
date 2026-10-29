"""``sage download``: the data an observing run or a run directory needs, into the data directory."""

import argparse
import os
from pathlib import Path

from sage.cli.registry import CommandError, closest, command
from sage.cli.training import on
from sage.config import machine

# Source: runs/<run>/dataset.py.
WORKERS = 16


def directory_needs(directory: Path, data_dir: Path) -> tuple:
    """
    What the run directory reads: per observing run its detectors and whether it reads the per-segment
    ASDs and the bank (the recolour target's, and those the fiducial pools), then the fiducial's runs and
    detectors. A run read only for the fiducial covers the detectors GWOSC lists for it.
    """
    from sage.data.download import run_detectors
    from sage.factory.build import asd_runs, bank_runs, strain_runs
    from sage.factory.train_run import check_run

    try:
        settings = check_run(directory)[0]
    except (ValueError, FileNotFoundError) as error:
        raise CommandError(str(error), f"edit {directory / 'run.yaml'} as above, then: {on('download', str(directory))}") from None
    noise, detectors = settings["noise"], settings["data"]["detectors"]
    strain, asds, banks = strain_runs(noise), asd_runs(noise), bank_runs(noise)
    fiducial = sorted({run.lower() for run in noise["fiducial"]})
    needs = {}
    for run in sorted({*strain, *banks, *fiducial}):
        read = detectors if run in strain or run in banks else [d for d in detectors if d in run_detectors(data_dir, run)]
        needs[run] = (read, run in asds, run in banks or run in fiducial)
    return needs, (fiducial, detectors)


def run_needs(run: str, detectors: list | None, data_dir: Path) -> dict:
    """The given detectors of an observing run, or every detector GWOSC lists for it, with its ASDs and bank."""
    from sage.data.download import gwosc_runs, recorded_gwosc, run_detectors

    if recorded_gwosc(data_dir, run.lower()) is None:
        names = gwosc_runs()
        if run.lower() not in names:
            match = closest(run.lower(), list(names))
            raise CommandError(f"observing run {run!r}: GWOSC lists {', '.join(sorted(names))}",
                               f"sage download {match}" if match else "sage download o3b")
    return {run.lower(): (detectors or run_detectors(data_dir, run.lower()), True, True)}


@command("download", "Download the data an observing run or a run directory needs", journey=3)
def download(arguments: list) -> None:
    """Download, build or adopt data in the data directory; each run's numbers are recorded beside it."""
    from requests.exceptions import RequestException

    parser = argparse.ArgumentParser(prog="sage download", description="Download the data an observing run or a run "
                                     "directory needs.")
    parser.add_argument("target", nargs="?", default=".", help="observing run (e.g. o3b) or run directory (default: here)")
    parser.add_argument("--detectors", nargs="+", help="with an observing run: these detectors only (default: all GWOSC lists)")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"parallel downloads (default: {WORKERS})")
    options = parser.parse_args(arguments)
    try:
        data_dir = machine.read_data_dir()
    except (FileNotFoundError, ValueError) as error:
        raise CommandError(str(error), "sage machine-setup") from None
    if not os.access(data_dir, os.W_OK | os.X_OK):
        raise CommandError(f"the data directory {data_dir} is not writable here", f"chmod u+rwx {data_dir}")

    try:
        fetch(options, data_dir)
    except RequestException as error:
        raise CommandError(f"GWOSC could not be reached from here: {error}",
                           f"once https://gwosc.org is reachable: sage download {options.target}") from None


def fetch(options: argparse.Namespace, data_dir: Path) -> None:
    """Download, build or adopt what ``options.target`` needs, printing each run's summary."""
    from sage.data.download import download_fiducial, download_run

    target = Path(options.target)
    if target.is_dir():
        if not (target / "run.yaml").exists():
            raise CommandError(f"no run.yaml in {target}", "sage init <directory> --preset <name>")
        if options.detectors:
            raise CommandError("--detectors applies to an observing run; a run directory reads its own",
                               on("download", options.target))
        needs, fiducial = directory_needs(target, data_dir)
    else:
        needs, fiducial = run_needs(options.target, options.detectors, data_dir), None

    for run, (detectors, asds, bank) in needs.items():
        try:
            summaries = download_run(data_dir, run, detectors, options.workers, asds, bank)
        except machine.MissingData as error:
            raise CommandError(str(error), f"sage download {options.target}") from None
        for detector, summary in summaries.items():
            first, last = summary["gps"]
            print(f"{run} {detector}: {summary['pieces']} pieces, {summary['days']:.2f} days, GPS {first:.0f}-{last:.0f}")
        print(f"Recorded in {machine.dataset_file(data_dir, run)}")
        if asds or bank:
            print(f"ASDs checked; recorded in {machine.asds_file(data_dir, run)}")
    if fiducial:
        print(f"Fiducial ASDs checked: {download_fiducial(data_dir, *fiducial)}")
    print(f"Next: {on('check', options.target) if target.is_dir() else 'sage check <run directory>'}")
