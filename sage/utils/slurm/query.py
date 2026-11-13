"""
Read-only SLURM queries: partitions, GPUs, accounts, QOS limits and a
job's output file.
"""

import getpass
import re
import subprocess

PARTITIONS = ["sinfo", "-h", "-o", "%P|%a|%l|%L|%D|%c|%m|%G|%C"]
NODE_GPUS = ["sinfo", "-h", "-N", "-O", "Partition:256,Gres:512,GresUsed:512"]
QOS_LIMITS = ["sacctmgr", "-n", "-P", "show", "qos", "format=name,maxwall,maxtrespu,maxjobspu"]
GPU = re.compile(r"gpu(?::([^:,(]+))?:(\d+)")


class Unavailable(Exception):
    """Query the site does not answer: missing, failed or timed out."""


def run(command: list) -> str:
    """
    Standard output of a read-only command, with a 30 s timeout.

    Parameters
    ----------
    command : list of str
        Command and arguments.

    Returns
    -------
    str

    Raises
    ------
    Unavailable
        If the command cannot run, times out or exits with an error.
    """
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Unavailable(f"{command[0]}: {error}") from None
    if done.returncode != 0:
        raise Unavailable(f"{command[0]}: {(done.stderr.strip() or 'failed').splitlines()[0]}")
    return done.stdout


def gpus(gres: str) -> dict:
    """
    GPU counts by type in a GRES string.

    ``gpu:h100:2(IDX:0-1)`` gives ``{"h100": 2}``; an untyped ``gpu:2``
    counts under ``"gpu"``.

    Parameters
    ----------
    gres : str
        GRES string from ``sinfo``.

    Returns
    -------
    dict of str to int
    """
    counts = {}
    for name, count in GPU.findall(gres):
        counts[name or "gpu"] = counts.get(name or "gpu", 0) + int(count)
    return counts


def parse_partitions(text: str) -> dict:
    """
    Partitions from ``sinfo`` output, one line per node type.

    Parameters
    ----------
    text : str
        Output of ``PARTITIONS``, format ``%P|%a|%l|%L|%D|%c|%m|%G|%C``.

    Returns
    -------
    dict
        Partition name to ``default`` (bool), ``state``, ``time_limit``,
        ``default_time``, ``cpus`` (allocated, idle, other, total) and
        ``node_types`` (nodes, CPUs, memory in MB, GPUs by type of
        each line).
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
    """
    GPUs in use and in total per partition, from per-node ``sinfo``.

    Parameters
    ----------
    text : str
        Output of ``NODE_GPUS``: partition, GRES and GRES used per node.

    Returns
    -------
    dict
        Partition name to ``(GPUs in use, GPUs in total)``.
    """
    usage = {}
    for line in text.splitlines():
        partition, gres, used = line.split()
        in_use, total = usage.get(partition.rstrip("*"), (0, 0))
        usage[partition.rstrip("*")] = (in_use + sum(gpus(used).values()), total + sum(gpus(gres).values()))
    return usage


def parse_qos_limits(text: str) -> dict:
    """
    QOS limits from ``sacctmgr show qos`` output.

    Parameters
    ----------
    text : str
        Output of ``QOS_LIMITS``.

    Returns
    -------
    dict
        QOS name to ``max_wall``, ``max_per_user`` (TRES) and
        ``max_jobs``, as strings.
    """
    limits = {}
    for line in text.splitlines():
        name, wall, per_user, jobs = line.split("|")
        limits[name] = {"max_wall": wall, "max_per_user": per_user, "max_jobs": jobs}
    return limits


def associations(user: str | None = None) -> list:
    """
    A user's associations, from ``sacctmgr show assoc``.

    Parameters
    ----------
    user : str or None
        User name; None for the current user.

    Returns
    -------
    list of tuple
        ``(account, partition, list of QOS)`` per association.

    Raises
    ------
    Unavailable
        If ``sacctmgr`` fails.
    """
    user = user or getpass.getuser()
    text = run(["sacctmgr", "-n", "-P", "show", "assoc", f"user={user}", "format=account,partition,qos"])
    return [(account, partition, qos.split(",") if qos else [])
            for account, partition, qos in (line.split("|") for line in text.splitlines())]


def job_output(job: str) -> str | None:
    """
    A job's standard-output file, from ``scontrol show job``.

    Parameters
    ----------
    job : str
        Job id.

    Returns
    -------
    str or None
        The ``StdOut`` path; None if the query fails or gives none.
    """
    try:
        fields = run(["scontrol", "show", "job", "-o", job]).split()
    except Unavailable:
        return None
    return next((field.removeprefix("StdOut=") for field in fields if field.startswith("StdOut=")), None)
