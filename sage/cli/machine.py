"""``sage setup`` and ``sage machine``: the machine file."""

import argparse
import os
from pathlib import Path

import yaml

from sage.cli.registry import CommandError, command
from sage.config import machine
from sage.utils.schedulers import schedulers
from sage.utils.slurm import query


@command(
    "setup",
    "Write the file that says where all data goes (once per machine)",
    journey=1,
)
def setup(arguments: list) -> None:
    """
    Write the machine file; ``data_dir`` is left empty unless given.

    An existing machine file is kept when ``--data-dir`` is not given.

    Parameters
    ----------
    arguments : list of str
        Command-line arguments after ``sage setup``.

    Raises
    ------
    CommandError
        If ``--data-dir`` is not a directory.
    """
    parser = argparse.ArgumentParser(
        prog="sage setup",
        description="Write the machine file: the directory for all data.",
    )
    parser.add_argument("--data-dir", help="data directory (default: fill it in yourself)")
    options = parser.parse_args(arguments)

    path = machine.MACHINE_FILE
    if options.data_dir is None:
        if path.exists():
            print(f"{path} exists; edit data_dir there, or: "
                  "sage setup --data-dir <directory>")
            print("Next: sage machine")
            return
        machine.write_machine_file(None)
        print(f"Wrote {path}; open it and fill in data_dir")
        print("Next: sage machine")
        return

    data_dir = Path(options.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise CommandError(
            f"{data_dir} is not a directory",
            f"mkdir -p {data_dir} && sage setup --data-dir {data_dir}",
        )
    machine.write_machine_file(data_dir)
    print(f"Wrote {path}: data_dir {data_dir}")
    print("Next: sage machine")


@command("machine", "Show the machine file and check its data directory")
def show_machine(arguments: list) -> None:
    """
    Print the machine file, its data directory and the scheduler.

    Parameters
    ----------
    arguments : list of str
        Command-line arguments after ``sage machine``.

    Raises
    ------
    CommandError
        If the machine file is missing or invalid, or its data
        directory cannot be read and written.
    """
    parser = argparse.ArgumentParser(prog="sage machine", description="Show the machine file and check its data directory.")
    parser.parse_args(arguments)

    if not machine.MACHINE_FILE.exists():
        raise CommandError(f"no machine file at {machine.MACHINE_FILE}", "sage setup")
    try:
        data_dir = machine.read_data_dir()
    except (FileNotFoundError, ValueError, yaml.YAMLError) as error:
        raise CommandError(str(error), "sage setup --data-dir <directory>") from None
    print(f"Machine file: {machine.MACHINE_FILE}")
    print(f"data_dir: {data_dir}")
    if not os.access(data_dir, os.R_OK | os.W_OK | os.X_OK):
        raise CommandError(f"data_dir {data_dir} is not readable and writable by you",
                           "sage setup --data-dir <directory you can write>")
    found = schedulers()
    if not found:
        print("Scheduler: none (no sbatch or condor_submit on PATH); jobs run where they are started")
    if "slurm" in found:
        print("\n".join(slurm_report()))
    if "htcondor" in found:
        print("Scheduler: HTCondor (its queries are not available yet)")


def qos_text(name: str, limits: dict) -> str:
    """
    A QOS and its limits, e.g. ``normal (wall 2-00:00:00, 500 jobs)``.

    Parameters
    ----------
    name : str
        QOS name.
    limits : dict
        QOS name -> ``max_wall``, ``max_per_user``, ``max_jobs``, as
        ``query.parse_qos_limits`` returns.

    Returns
    -------
    str
        The name, followed by its non-empty limits if it has any.
    """
    if name not in limits:
        return name
    entry = limits[name]
    parts = [f"wall {entry['max_wall']}" if entry["max_wall"] else "",
             f"per user {entry['max_per_user']}" if entry["max_per_user"] else "",
             f"{entry['max_jobs']} jobs" if entry["max_jobs"] else ""]
    return f"{name} ({', '.join(p for p in parts if p)})" if any(parts) else name


def slurm_report() -> list:
    """
    Lines on the SLURM partitions, accounts and QOS of this cluster.

    A query the site refuses is reported as unavailable.

    Returns
    -------
    list of str
        Report lines.
    """
    lines = ["Scheduler: SLURM"]
    try:
        partitions = query.parse_partitions(query.run(query.PARTITIONS))
    except query.Unavailable as error:
        return lines + [f"Partitions: unavailable ({error})"]
    try:
        gpu_use, gpu_note = query.parse_node_gpus(query.run(query.NODE_GPUS)), ""
    except query.Unavailable as error:
        gpu_use, gpu_note = {}, f" (GPU use unavailable: {error})"
    lines.append(f"Partitions (* default){gpu_note}:")
    for name, entry in partitions.items():
        allocated, idle, other, total = entry["cpus"]
        nodes = "; ".join(
            f"{t['nodes']} x {t['cpus']} CPUs, {t['memory_mb']} MB"
            + (", GPUs " + " ".join(f"{gpu}:{count}" for gpu, count in t["gpus"].items()) if t["gpus"] else "")
            for t in entry["node_types"]
        )
        in_use, gpu_total = gpu_use.get(name, (0, 0))
        gpus = f", GPUs in use {in_use}/{gpu_total}" if gpu_total else ""
        lines.append(f"  {name}{'*' if entry['default'] else ''}: {entry['state']}, time limit {entry['time_limit']} "
                     f"(default {entry['default_time']}); {nodes}; CPUs idle {idle}/{total}{gpus}")
    try:
        associations = query.associations()
        limits = query.parse_qos_limits(query.run(query.QOS_LIMITS))
    except query.Unavailable as error:
        return lines + [f"Accounts and QOS: unavailable ({error})"]
    for account, partition, qos in associations:
        where = f" on {partition}" if partition else ""
        lines.append(f"Account {account}{where}: QOS {', '.join(qos_text(q, limits) for q in qos) or 'none'}")
    return lines
