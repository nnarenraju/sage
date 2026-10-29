"""``sage machine-setup`` and ``sage machine``: the machine file."""

import argparse
import os
from pathlib import Path

import yaml

from sage.cli.journey import next_command
from sage.cli.registry import CommandError, command
from sage.config import machine
from sage.config.loader import check, read_yaml
from sage.utils.schedulers import schedulers
from sage.utils.slurm import query


@command("machine-setup", "Tell Sage where all downloaded data goes (once per machine)", journey=1)
def setup_machine(arguments: list) -> None:
    """Write the machine file, asking for the data directory unless given."""
    parser = argparse.ArgumentParser(
        prog="sage machine-setup", description="Write the machine file: the directory all downloaded data goes to."
    )
    parser.add_argument("--data-dir", help="data directory (asked for if not given)")
    options = parser.parse_args(arguments)

    current = None
    if machine.MACHINE_FILE.exists():
        try:
            current = check(read_yaml(machine.MACHINE_FILE), machine.SCHEMA, str(machine.MACHINE_FILE))["data_dir"]
        except (ValueError, yaml.YAMLError) as error:
            print(f"{machine.MACHINE_FILE} is unreadable and will be replaced: {error}")
    answer = options.data_dir
    if answer is None:
        answer = input(f"Directory for all downloaded data{f' [{current}]' if current else ''}: ").strip() or current
    if not answer:
        raise CommandError("no data directory given", "sage machine-setup --data-dir <directory>")

    data_dir = Path(answer).expanduser().resolve()
    if not data_dir.is_dir():
        raise CommandError(f"{data_dir} is not a directory", f"mkdir -p {data_dir} && sage machine-setup --data-dir {data_dir}")
    machine.write_data_dir(data_dir)
    print(f"Wrote {machine.MACHINE_FILE}: data_dir {data_dir}")
    print("Next: sage machine")


@command("machine", "Show the machine file and check its data directory")
def show_machine(arguments: list) -> None:
    """Print the machine file and check that its data directory can be read and written."""
    parser = argparse.ArgumentParser(prog="sage machine", description="Show the machine file and check its data directory.")
    parser.parse_args(arguments)

    if not machine.MACHINE_FILE.exists():
        raise CommandError(f"no machine file at {machine.MACHINE_FILE}", "sage machine-setup")
    try:
        data_dir = machine.read_data_dir()
    except (FileNotFoundError, ValueError, yaml.YAMLError) as error:
        raise CommandError(str(error), "sage machine-setup") from None
    print(f"Machine file: {machine.MACHINE_FILE}")
    print(f"data_dir: {data_dir}")
    if not os.access(data_dir, os.R_OK | os.W_OK | os.X_OK):
        raise CommandError(f"data_dir {data_dir} is not readable and writable by you",
                           "sage machine-setup --data-dir <directory you can write>")
    found = schedulers()
    if not found:
        print("Scheduler: none (no sbatch or condor_submit on PATH); jobs run where they are started")
    if "slurm" in found:
        print("\n".join(slurm_report()))
    if "htcondor" in found:
        print("Scheduler: HTCondor (its queries are not available yet)")
    print(f"Next: {next_command()}")


def qos_text(name: str, limits: dict) -> str:
    """A QOS and its limits, e.g. ``normal (wall 2-00:00:00, per user cpu=10000, 500 jobs)``."""
    if name not in limits:
        return name
    entry = limits[name]
    parts = [f"wall {entry['max_wall']}" if entry["max_wall"] else "",
             f"per user {entry['max_per_user']}" if entry["max_per_user"] else "",
             f"{entry['max_jobs']} jobs" if entry["max_jobs"] else ""]
    return f"{name} ({', '.join(p for p in parts if p)})" if any(parts) else name


def slurm_report() -> list:
    """Lines on what the SLURM cluster offers; a query the site refuses is reported as unavailable."""
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
