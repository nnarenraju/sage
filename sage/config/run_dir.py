"""
Writing and checking the run directory record under ``resolved/``.

``run.yaml``, ``prior.yaml``, ``derived.yaml``, ``versions.yaml`` (one
entry per job segment) and ``fiducial/``. Once ``checkpoints/`` holds a
checkpoint, the settings, prior and derived values may not change.
"""

import datetime
import os
import platform
import shutil
import subprocess
from pathlib import Path

import torch
import yaml

from sage.config.loader import load_run, yaml_text
from sage.config.machine import fiducial_file
from sage.config.prior import load_prior_file

REPO = Path(__file__).resolve().parents[2]


class SettingsChanged(ValueError):
    """Settings, prior or derived values changed once training began."""


class NotPrepared(Exception):
    """Before training, ``resolved/`` is missing or out of date."""


class StateMismatch(ValueError):
    """
    Settings differ from a checkpoint's in a way its state cannot take.

    Parameters
    ----------
    message : str
        The changes.
    weights_only : bool
        Whether a weights-only clone takes them.
    """

    def __init__(self, message: str, weights_only: bool):
        super().__init__(message)
        self.weights_only = weights_only


def find_run_dir(directory: str | Path) -> Path:
    """
    Nearest directory at or above ``directory`` holding a ``run.yaml``.

    Found as git finds ``.git``.

    Parameters
    ----------
    directory : str or Path
        Where to start, e.g. ``"."``.

    Returns
    -------
    Path
        ``directory`` as given if it holds a ``run.yaml`` or no
        directory above it does; otherwise the absolute path of the
        nearest directory above it that holds one.
    """
    start = Path(directory)
    if (start / "run.yaml").exists():
        return start
    for parent in Path(os.path.abspath(start)).parents:
        if (parent / "run.yaml").exists():
            return parent
    return start


def training_started(run_dir: Path) -> bool:
    """
    Whether ``checkpoints/`` holds a checkpoint.

    Parameters
    ----------
    run_dir : Path
        Run directory.

    Returns
    -------
    bool
    """
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
    Write a mapping as block YAML with scalar lists inline.

    Parameters
    ----------
    path : Path
        Output file; its directory is created.
    values : dict
        Mapping to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_text(values))


def check_record(run_dir: Path, settings: dict, prior: dict, resolve) -> None:
    """
    Check that a started run's settings and prior match the record.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    settings : dict
        Resolved settings, ``resolve(load_run(...))``.
    prior : dict
        Checked prior, from ``load_prior``.
    resolve : callable
        Fills a loaded run file's components
        (``build.resolve_settings``).

    Raises
    ------
    SettingsChanged
        If the settings or prior differ from the record.
    ValueError
        If ``resolved/run.yaml`` or ``resolved/prior.yaml`` is missing.
    """
    run_dir = Path(run_dir)
    records = {"run.yaml": (settings, lambda path: resolve(load_run(path))), "prior.yaml": (prior, load_prior_file)}
    for name, (values, load) in records.items():
        path = run_dir / "resolved" / name
        if not path.exists():
            raise ValueError(f"{run_dir}: checkpoints exist but resolved/{name} is missing")
        changed = differences(load(path), values)
        if changed:
            raise SettingsChanged(
                f"{run_dir}: resolved/{name} differs after training started "
                f"({'; '.join(changed)}); the changed settings continue in a clone of this run"
            )


def prepare_run_dir(run_dir: Path, settings: dict, prior: dict, resolve) -> None:
    """
    Record the resolved settings and prior, or check them once started.

    Before training starts the record is rewritten; after, both must
    resolve exactly as recorded (``check_record``).

    Parameters
    ----------
    run_dir : Path
        Run directory.
    settings : dict
        Resolved settings, ``resolve(load_run(...))``.
    prior : dict
        Checked prior, from ``load_prior``.
    resolve : callable
        Fills a loaded run file's components
        (``build.resolve_settings``).

    Raises
    ------
    SettingsChanged
        If training has started and the settings or prior differ from
        the record.
    ValueError
        If checkpoints exist without a record.
    """
    run_dir = Path(run_dir)
    if training_started(run_dir):
        check_record(run_dir, settings, prior, resolve)
        return
    write_yaml(run_dir / "resolved" / "run.yaml", settings)
    write_yaml(run_dir / "resolved" / "prior.yaml", prior)


def check_prepared(run_dir: Path, settings: dict, prior: dict, resolve) -> None:
    """
    Check that ``resolved/`` is the current record of the run.

    Before training starts it must exist and match ``run.yaml`` and the
    prior (``sage prepare``); after, ``check_record`` applies.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    settings : dict
        Resolved settings, ``resolve(load_run(...))``.
    prior : dict
        Checked prior, from ``load_prior``.
    resolve : callable
        Fills a loaded run file's components
        (``build.resolve_settings``).

    Raises
    ------
    NotPrepared
        If training has not started and the record is missing or
        differs.
    SettingsChanged
        If training has started and the settings or prior differ.
    ValueError
        If training has started and the record is missing.
    """
    run_dir = Path(run_dir)
    if training_started(run_dir):
        check_record(run_dir, settings, prior, resolve)
        return
    for name in ("run.yaml", "prior.yaml", "derived.yaml", "fiducial"):
        if not (run_dir / "resolved" / name).exists():
            raise NotPrepared(f"{run_dir}: not prepared (resolved/{name} is missing)")
    records = {"run.yaml": (settings, lambda path: resolve(load_run(path))), "prior.yaml": (prior, load_prior_file)}
    for name, (values, load) in records.items():
        changed = differences(load(run_dir / "resolved" / name), values)
        if changed:
            source = "run.yaml" if name == "run.yaml" else "the prior"
            raise NotPrepared(f"{run_dir}: {source} changed since sage prepare ({'; '.join(changed)})")


def check_derived(run_dir: Path, derived: dict) -> None:
    """
    Check that the run's derived values equal ``resolved/derived.yaml``.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    derived : dict
        Values from ``train_run.derived_values``.

    Raises
    ------
    NotPrepared
        If training has not started and the values differ.
    SettingsChanged
        If training has started and the values differ.
    FileNotFoundError
        If ``resolved/derived.yaml`` is missing.
    """
    changed = differences(yaml.safe_load((Path(run_dir) / "resolved" / "derived.yaml").read_text()), derived)
    if changed and not training_started(run_dir):
        raise NotPrepared(f"{run_dir}: derived values differ from those sage prepare wrote ({'; '.join(changed)})")
    if changed:
        raise SettingsChanged(
            f"{run_dir}: derived values differ after training started ({'; '.join(changed)}); "
            "the network input or scaling would change mid-run"
        )


def record_derived(run_dir: Path, derived: dict) -> None:
    """
    Record the derived values, or check them once training has started.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    derived : dict
        Window geometry, multirate bins, target scaling, flat-tail
        start epoch and seeds.

    Raises
    ------
    SettingsChanged
        If training has started and the derived values differ from the
        record.
    FileNotFoundError
        If training has started and ``resolved/derived.yaml`` is
        missing.
    """
    if training_started(run_dir):
        check_derived(run_dir, derived)
        return
    write_yaml(Path(run_dir) / "resolved" / "derived.yaml", derived)


def git(*args) -> str:
    """
    Output of a git command in the Sage checkout.

    Parameters
    ----------
    *args : str
        Git arguments.

    Returns
    -------
    str
        Standard output.

    Raises
    ------
    subprocess.CalledProcessError
        If git exits with an error.
    OSError
        If git cannot be run.
    """
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def code_differs(commit: str) -> bool:
    """
    Whether the code under ``sage/`` differs from ``commit``.

    Differences are any committed or uncommitted edits, and untracked
    files that git does not ignore.

    Parameters
    ----------
    commit : str
        Commit to compare with.

    Returns
    -------
    bool
    """
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
    """
    Python, PyTorch, CUDA, PyCBC and LALSuite versions, and the device.

    Parameters
    ----------
    device : str or torch.device
        Training device.

    Returns
    -------
    dict
        ``python``, ``torch``, ``cuda``, ``pycbc``, ``lalsuite`` and
        ``device`` (the GPU name for CUDA, else the device type).
    """
    # Deferred: heavy imports used only here
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


def record_versions(run_dir: Path, start_epoch: int, device) -> None:
    """
    Append this job segment to ``resolved/versions.yaml``.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    start_epoch : int
        Epoch this segment starts at.
    device : str or torch.device
        Training device.

    Raises
    ------
    RuntimeError
        If Sage is not running from a git checkout.
    """
    path = Path(run_dir) / "resolved" / "versions.yaml"
    segments = yaml.safe_load(path.read_text())["segments"] if path.exists() else []
    version = code_version()
    segments.append(
        {
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "start_epoch": start_epoch,
            "sage_commit": version["commit"],
            "sage_uncommitted_changes": version["uncommitted_changes"],
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
            name = fiducial_file(fiducial_dir, detector).with_suffix(suffix).name
            shutil.copy2(Path(fiducial_dir) / name, copy / name)
    return copy
