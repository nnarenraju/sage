#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        : servers.py
Description     : Single source of truth for per-server settings.

This module centralises everything that changes when Sage is run on a
different machine: where data is saved, where the conda env lives, and how
batch jobs are launched. A run is wired to a server with one switch -- the
``SAGE_SERVER`` environment variable (or hostname auto-detection) -- and every
path, both in Python (``config.py`` / ``dataset.py``) and in the shell run
scripts (``sage/utils/run_base.sh``), is derived from the same entry here.

Add a new machine by adding one ``Server`` to ``SERVERS`` below.

Shell side
----------
The shell helpers read these same values via::

    eval "$(python -m sage.utils.servers env)"

so there is no duplicated path list to keep in sync.

CLI
---
    python -m sage.utils.servers name     # resolved server name
    python -m sage.utils.servers show     # human-readable dump
    python -m sage.utils.servers env      # `export KEY=VAL` lines for the shell
"""

from __future__ import annotations

import os
import sys
import socket
import fnmatch
from pathlib import Path
from dataclasses import dataclass, field, fields

# Repo root = .../sage  (this file is .../sage/sage/utils/servers.py)
REPO_ROOT = str(Path(__file__).resolve().parents[2])


@dataclass(frozen=True)
class Server:
    """Everything that is specific to one machine.

    Paths
    -----
    home        : the user's home directory on that machine.
    data_root   : parent dir under which ``data_release/`` is created. This is
                  the big-storage mount where downloaded noise lands and from
                  which training reads. Mirrors ``save_parent_dir`` of
                  :class:`sage.data.primer.DataReleaseDownloader`.
    work_root   : scratch / cache root (model compile caches, torch hub, etc.).
    repo_root   : location of the sage checkout. Defaults to this checkout.

    Launch
    ------
    python      : absolute path to the interpreter (conda env) for jobs. If
                  ``None`` the shell falls back to whatever ``python`` is on PATH.
    scheduler   : "slurm" or "local".
    partition / qos / gres / time / cpus / mem / account / mail :
                  SLURM ``sbatch`` parameters (ignored when scheduler="local").
    """

    name: str
    hostmatch: tuple[str, ...]            # fnmatch patterns against fqdn/hostname

    home: str
    data_root: str
    work_root: str
    repo_root: str = REPO_ROOT

    python: str | None = None
    scheduler: str = "local"             # "slurm" | "local"
    partition: str = ""
    qos: str = ""
    gres: str = ""
    time: str = ""
    cpus: int = 4
    mem: str = ""
    account: str = ""
    mail: str = ""

    # ----- derived data-release layout ------------------------------------
    @property
    def data_release_root(self) -> str:
        return os.path.join(self.data_root, "data_release")

    def dataset_dir(self, run: str) -> str:
        """e.g. <data_root>/data_release/o3b_dataset"""
        return os.path.join(self.data_release_root, f"{run.lower()}_dataset")

    def data_dir(self, run: str) -> str:
        """The per-run ``data_dir`` (recolour ASD banks etc.)."""
        return os.path.join(self.dataset_dir(run), "data_dir")

    def noise_bin(self, detector: str, run: str,
                  release_dirname: str = "data_release") -> str:
        """Monolithic noise .bin for one detector / observing run.

        The downloader writes the .bin FLAT inside its release folder with the
        run encoded in the filename -- there is NO ``<run>_dataset`` sub-folder
        for the strain (that sub-folder only holds derived ASD banks).
        ``release_dirname`` is the folder the run was downloaded into:
        ``"data_release"`` (the default/base run) or an isolated dir such as
        ``"data_release_o3a"`` / ``"data_release_o4b"``.
        """
        return os.path.join(
            self.data_root, release_dirname, f"data_{detector}_{run}.bin"
        )

    def run_noise(self, run: str, detectors, release_dirname: str = "data_release",
                  data_dir: str | None = None) -> dict:
        """One run's entry for the multi-run noise sampler's ``training_noise``.

        Bundles the per-detector strain ``.bin`` (from ``release_dirname``) with
        the run's derived-ASD ``data_dir`` (holding the per-segment whitening
        ASDs the recolour step needs). ``bins`` are ordered to match
        ``detectors``. Pass ``data_dir`` explicitly for the flat O4 layout
        (``data_release_<run>/data_dir``); it defaults to the O3-style
        ``data_release/<run>_dataset/data_dir``.
        """
        return {
            "run": run,
            "data_dir": data_dir if data_dir is not None else self.data_dir(run),
            "bins": [self.noise_bin(d, run, release_dirname) for d in detectors],
            "detectors": list(detectors),  # validated against cfg.detectors by the sampler
        }


# ----------------------------------------------------------------------------
# Server registry -- add a machine here and nothing else needs to change.
# ----------------------------------------------------------------------------
SERVERS: dict[str, Server] = {
    # Potsdam HPC (current). SLURM, big storage on /work.
    "jarvis": Server(
        name="jarvis",
        hostmatch=("*.hpc.uni-potsdam.de", "jlogin*"),
        home="/home/nagarajan",
        data_root="/work/nagarajan",
        work_root="/work/nagarajan",
        python="/home/nagarajan/.conda/envs/sage/bin/python",
        scheduler="slurm",
        partition="gpu",
        qos="normal",
        gres="gpu:1",     # any GPU type (hgc01 h100_80gb + hgc02 h100_94gb/h200_143gb);
                          # all fit the model -- pinning to h100_80gb starved us to 2 GPUs
        time="2-00:00",
        cpus=16,          # noise prefetch (16 read workers) + hard-mining threads
        # RecolourPostprocess keeps the target ASD bank resident: ~16 GB/detector
        # (see recolour.py). 2-det ~33 GB, 3-det (HLV) ~49 GB, + segment banks +
        # torch/compile overhead. 128 GB covers all networks; nodes have ~770 GB.
        mem="128G",
        mail="nagarajan@uni-potsdam.de",
    ),

    # Glasgow WIAY. Fill SLURM details if/when batch jobs are run there.
    "wiay": Server(
        name="wiay",
        hostmatch=("*.wiay.*", "*wiay*"),
        home="/home/nnarenraju",
        data_root="/data/wiay/nnarenraju",
        work_root="/data/wiay/nnarenraju",
        python=None,
        scheduler="local",
    ),

    # IGR local-scratch box.
    "scotdist": Server(
        name="scotdist",
        hostmatch=("*scotdist*",),
        home="/home/nnarenraju",
        data_root="/local/scratch/igr/nnarenraju",
        work_root="/local/scratch/igr/nnarenraju",
        python=None,
        scheduler="local",
    ),
}


# ----------------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------------
def _hostnames() -> list[str]:
    names = []
    try:
        names.append(socket.getfqdn())
    except Exception:
        pass
    try:
        names.append(socket.gethostname())
    except Exception:
        pass
    return [n for n in names if n]


def _detect_by_hostname() -> Server | None:
    hosts = _hostnames()
    for srv in SERVERS.values():
        for host in hosts:
            if any(fnmatch.fnmatch(host, pat) for pat in srv.hostmatch):
                return srv
    return None


def get_server(name: str | None = None) -> Server:
    """Resolve the active server.

    Order: explicit ``name`` arg -> ``$SAGE_SERVER`` -> hostname auto-detect.
    Raises a clear error (listing valid names and the current hostname) if it
    cannot be determined, so a run never silently writes to the wrong place.
    """
    name = name or os.environ.get("SAGE_SERVER")
    if name:
        try:
            return SERVERS[name]
        except KeyError:
            raise KeyError(
                f"Unknown server '{name}'. Known servers: {sorted(SERVERS)}. "
                "Set SAGE_SERVER to one of these or add an entry in "
                "sage/utils/servers.py."
            )
    detected = _detect_by_hostname()
    if detected is not None:
        return detected
    raise RuntimeError(
        "Could not determine the active server. Set the SAGE_SERVER "
        f"environment variable to one of {sorted(SERVERS)}. "
        f"(hostname={_hostnames()})"
    )


# ----------------------------------------------------------------------------
# Shell bridge:  python -m sage.utils.servers env
# ----------------------------------------------------------------------------
def _shell_exports(srv: Server) -> str:
    """Emit `export KEY=VAL` lines so run_base.sh inherits the same values."""
    def q(v) -> str:
        return '"' + str(v).replace('"', '\\"') + '"'

    out = {
        "SAGE_SERVER": srv.name,
        "SAGE_HOME": srv.home,
        "SAGE_DATA_ROOT": srv.data_root,
        "SAGE_DATA_RELEASE": srv.data_release_root,
        "SAGE_WORK_ROOT": srv.work_root,
        "SAGE_REPO_ROOT": srv.repo_root,
        "SAGE_PYTHON": srv.python or "",
        "SAGE_SCHEDULER": srv.scheduler,
        "SAGE_PARTITION": srv.partition,
        "SAGE_QOS": srv.qos,
        "SAGE_GRES": srv.gres,
        "SAGE_TIME": srv.time,
        "SAGE_CPUS": srv.cpus,
        "SAGE_MEM": srv.mem,
        "SAGE_ACCOUNT": srv.account,
        "SAGE_MAIL": srv.mail,
    }
    return "\n".join(f"export {k}={q(v)}" for k, v in out.items())


def _human(srv: Server) -> str:
    lines = [f"Active server: {srv.name}"]
    for f in fields(srv):
        lines.append(f"  {f.name:12s} = {getattr(srv, f.name)}")
    lines.append(f"  data_release_root = {srv.data_release_root}")
    return "\n".join(lines)


def _main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "show"
    # Allow `... env potsdam` to force a server without exporting first.
    forced = argv[2] if len(argv) > 2 else None
    srv = get_server(forced)
    if cmd == "name":
        print(srv.name)
    elif cmd == "env":
        print(_shell_exports(srv))
    elif cmd == "show":
        print(_human(srv))
    else:
        print(f"Unknown command '{cmd}'. Use: name | show | env", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
