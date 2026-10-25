"""The run directory record, under ``resolved/``.

``run.yaml`` (every setting used, defaults included), ``prior.yaml``,
``derived.yaml`` (values computed from them), ``versions.yaml`` (code and
libraries, one entry per job segment) and ``fiducial/`` (the ASDs used).
Training counts as started once ``checkpoints/`` holds a checkpoint; from
then on the settings, prior and derived values may not change, and code that
differs from the last segment's commit needs ``--accept-code-change``.
"""

import datetime
import platform
import shutil
import subprocess
from pathlib import Path

import torch
import yaml

from sage.config.loader import load_run
from sage.config.prior import load_prior_file

REPO = Path(__file__).resolve().parents[2]


def training_started(run_dir: Path) -> bool:
    """Whether ``checkpoints/`` holds a checkpoint."""
    return any((Path(run_dir) / "checkpoints").glob("*.pt"))


def differences(old: dict, new: dict, where: str = "") -> list:
    """
    List every entry that differs between two resolved mappings.

    Parameters
    ----------
    old, new : dict
        Resolved settings or priors.
    where : str
        Dotted location of these mappings; empty at the top.

    Returns
    -------
    list of str
        One ``location: old -> new`` entry per difference.
    """
    found = []
    for key in sorted(set(old) | set(new), key=str):
        location = f"{where}.{key}" if where else str(key)
        before, after = old.get(key, "<absent>"), new.get(key, "<absent>")
        if isinstance(before, dict) and isinstance(after, dict):
            found += differences(before, after, location)
        elif before != after:
            found.append(f"{location}: {before!r} -> {after!r}")
    return found


def write_yaml(path: Path, values: dict) -> None:
    """
    Write a mapping as block YAML with short scalar lists inline.

    Parameters
    ----------
    path : Path
        Output file; its directory is created.
    values : dict
        Mapping to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        yaml.safe_dump(values, file, sort_keys=False, default_flow_style=None)


def prepare_run_dir(run_dir: Path, settings: dict, prior: dict, resolve) -> None:
    """
    Record the resolved settings and prior, or check them against the record.

    Before training starts the record is rewritten; after, both must
    resolve exactly as recorded.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    settings : dict
        Resolved settings, ``resolve(load_run(...))``.
    prior : dict
        Checked prior, from ``load_prior``.
    resolve : callable
        Fills a loaded run file's components (``build.resolve_settings``).

    Raises
    ------
    ValueError
        If training has started and the settings or prior differ from the
        record, or checkpoints exist without a record.
    """
    run_dir = Path(run_dir)
    records = {
        "run.yaml": (settings, lambda path: resolve(load_run(path))),
        "prior.yaml": (prior, load_prior_file),
    }
    if not training_started(run_dir):
        for name, (values, _) in records.items():
            write_yaml(run_dir / "resolved" / name, values)
        return

    for name, (values, load) in records.items():
        path = run_dir / "resolved" / name
        if not path.exists():
            raise ValueError(f"{run_dir}: checkpoints exist but resolved/{name} is missing")
        changed = differences(load(path), values)
        if changed:
            raise ValueError(
                f"{run_dir}: resolved/{name} differs after training started "
                f"({'; '.join(changed)}); use a new run directory"
            )


def record_derived(run_dir: Path, derived: dict) -> None:
    """
    Record the values derived from the settings, or check them on resume.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    derived : dict
        Window geometry, multirate bins, target scaling, flat-tail start,
        stream seeds.

    Raises
    ------
    ValueError
        If training has started and the derived values differ from the record.
    """
    path = Path(run_dir) / "resolved" / "derived.yaml"
    if not training_started(run_dir):
        write_yaml(path, derived)
        return
    changed = differences(yaml.safe_load(path.read_text()), derived)
    if changed:
        raise ValueError(
            f"{run_dir}: derived values differ after training started ({'; '.join(changed)}); "
            "the network input or scaling would change mid-run"
        )


def git(*args) -> str:
    """Output of a git command in the Sage checkout."""
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def code_differs(commit: str) -> bool:
    """Whether the code under ``sage/`` differs from ``commit`` in any way
    (other commits, uncommitted edits or new files)."""
    diff = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", commit, "--", "sage"], capture_output=True)
    return diff.returncode != 0 or bool(git("ls-files", "--others", "--exclude-standard", "--", "sage").strip())


def code_version() -> dict:
    """
    The Sage commit and whether ``sage/`` holds uncommitted changes.

    Returns
    -------
    dict
        ``commit``, ``uncommitted_changes``.

    Raises
    ------
    RuntimeError
        If Sage is not running from a git checkout.
    """
    try:
        commit = git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot read the Sage version with git in {REPO}: {error}") from None
    return {"commit": commit, "uncommitted_changes": code_differs(commit)}


def environment(device) -> dict:
    """Python, PyTorch, CUDA, PyCBC and LALSuite versions and the device in use."""
    import lal
    import pycbc

    device = torch.device(device)
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "pycbc": str(pycbc.__version__),
        "lalsuite": str(lal.__version__),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
    }


def record_versions(run_dir: Path, start_epoch: int, device, accept_code_change: bool) -> None:
    """
    Append this job segment to ``resolved/versions.yaml``.

    Once training has started, code under ``sage/`` that differs in any way
    from the last segment's commit stops the run, unless
    ``accept_code_change``.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    start_epoch : int
        Epoch this segment starts at.
    device : str or torch.device
        Training device.
    accept_code_change : bool
        Continue on changed code.

    Raises
    ------
    ValueError
        If training has started, the code differs from the last segment's
        commit and ``accept_code_change`` is not set.
    """
    path = Path(run_dir) / "resolved" / "versions.yaml"
    segments = yaml.safe_load(path.read_text())["segments"] if path.exists() else []
    version = code_version()
    changed = bool(segments) and training_started(run_dir) and code_differs(segments[-1]["sage_commit"])
    if changed and not accept_code_change:
        raise ValueError(
            f"{run_dir}: the Sage code differs from commit {segments[-1]['sage_commit']}, which this run last "
            "trained on. To continue on the current code, run again with --accept-code-change"
        )
    segments.append(
        {
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "start_epoch": start_epoch,
            "sage_commit": version["commit"],
            "sage_uncommitted_changes": version["uncommitted_changes"],
            "code_change_accepted": changed,
            **environment(device),
        }
    )
    write_yaml(path, {"segments": segments})


def record_fiducial(run_dir: Path, fiducial_dir: Path, detectors: list) -> Path:
    """
    Copy the fiducial ASDs into the run before training starts.

    From the first checkpoint on, the copy is what the run reads.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    fiducial_dir : Path
        Fiducial directory under the data directory.
    detectors : list of str
        Detectors whose files are copied.

    Returns
    -------
    Path
        ``resolved/fiducial``.

    Raises
    ------
    ValueError
        If checkpoints exist but the copy does not.
    """
    copy = Path(run_dir) / "resolved" / "fiducial"
    if training_started(run_dir):
        if not copy.exists():
            raise ValueError(f"{run_dir}: checkpoints exist but resolved/fiducial is missing")
        return copy
    copy.mkdir(parents=True, exist_ok=True)
    for detector in detectors:
        for suffix in (".bin", ".json"):
            name = f"fiducial_{detector}_psd{suffix}"
            shutil.copy2(Path(fiducial_dir) / name, copy / name)
    return copy
