"""Read-only SLURM queries: partitions, GPUs, accounts, QOS limits and a job's output file."""

import getpass
import re
import subprocess

PARTITIONS = ["sinfo", "-h", "-o", "%P|%a|%l|%L|%D|%c|%m|%G|%C"]
NODE_GPUS = ["sinfo", "-h", "-N", "-O", "Partition:256,Gres:512,GresUsed:512"]
QOS_LIMITS = ["sacctmgr", "-n", "-P", "show", "qos", "format=name,maxwall,maxtrespu,maxjobspu"]
GPU = re.compile(r"gpu(?::([^:,(]+))?:(\d+)")


class Unavailable(Exception):
    """A query the site does not answer (command missing, refused or timed out)."""


def run(command: list) -> str:
    """Output of a read-only command; Unavailable if it fails."""
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Unavailable(f"{command[0]}: {error}") from None
    if done.returncode != 0:
        raise Unavailable(f"{command[0]}: {(done.stderr.strip() or 'failed').splitlines()[0]}")
    return done.stdout


def gpus(gres: str) -> dict:
    """GPU counts by type in a GRES string, e.g. ``gpu:h100:2(IDX:0-1)`` -> ``{"h100": 2}``."""
    counts = {}
    for name, count in GPU.findall(gres):
        counts[name or "gpu"] = counts.get(name or "gpu", 0) + int(count)
    return counts


def parse_partitions(text: str) -> dict:
    """
    Partitions from ``sinfo -o "%P|%a|%l|%L|%D|%c|%m|%G|%C"``, one line per node type.

    Returns
    -------
    dict
        Partition name to ``default`` (bool), ``state``, ``time_limit``,
        ``default_time``, ``cpus`` (allocated, idle, other, total) and
        ``node_types`` (list of nodes, CPUs, memory in MB, GPUs by type).
    """
    partitions = {}
    for line in text.splitlines():
        name, state, limit, default_time, nodes, cpus, memory, gres, cpu_state = line.split("|")
        entry = partitions.setdefault(name.rstrip("*"), {
            "default": name.endswith("*"), "state": state, "time_limit": limit, "default_time": default_time,
            "cpus": [0, 0, 0, 0], "node_types": [],
        })
        entry["cpus"] = [a + int(b) for a, b in zip(entry["cpus"], cpu_state.split("/"))]
        entry["node_types"].append({"nodes": int(nodes), "cpus": cpus, "memory_mb": memory, "gpus": gpus(gres)})
    return partitions


def parse_node_gpus(text: str) -> dict:
    """Partition to (GPUs in use, GPUs in total) from ``sinfo -N -O Partition,Gres,GresUsed``."""
    usage = {}
    for line in text.splitlines():
        partition, gres, used = line.split()
        in_use, total = usage.get(partition.rstrip("*"), (0, 0))
        usage[partition.rstrip("*")] = (in_use + sum(gpus(used).values()), total + sum(gpus(gres).values()))
    return usage


def parse_qos_limits(text: str) -> dict:
    """QOS name to ``max_wall``, ``max_per_user`` (TRES) and ``max_jobs`` from ``sacctmgr show qos``."""
    limits = {}
    for line in text.splitlines():
        name, wall, per_user, jobs = line.split("|")
        limits[name] = {"max_wall": wall, "max_per_user": per_user, "max_jobs": jobs}
    return limits


def associations(user: str | None = None) -> list:
    """The user's (account, partition, QOS list) associations, from ``sacctmgr``."""
    user = user or getpass.getuser()
    text = run(["sacctmgr", "-n", "-P", "show", "assoc", f"user={user}", "format=account,partition,qos"])
    return [(account, partition, qos.split(",") if qos else [])
            for account, partition, qos in (line.split("|") for line in text.splitlines())]


def job_output(job: str) -> str | None:
    """A job's standard-output file, from ``scontrol show job``; None if the query fails."""
    try:
        fields = run(["scontrol", "show", "job", "-o", job]).split()
    except Unavailable:
        return None
    return next((field.removeprefix("StdOut=") for field in fields if field.startswith("StdOut=")), None)
