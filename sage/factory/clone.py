"""Start a run directory from a stored checkpoint of another run (``sage clone``).

The new directory gets the old ``run.yaml`` (and a prior file it names), the
checkpoint it starts from under ``clone/``, and ``resolved/clone.yaml``.
"""

import dataclasses
import functools
import json
import operator
import os
import shutil
from pathlib import Path

import h5py
import torch
import yaml

from sage.config.loader import load_run_record
from sage.config.run_dir import StateMismatch, differences, training_started, write_yaml
from sage.factory.build import build_data, resolve_settings
from sage.utils.checkpoint import TRAINING_ONLY, next_epoch

# Settings the saved weights cannot follow a change of; with full state also
# those the restored optimiser and schedule states would override.
BREAKING = ("model", "data.detectors", "signal.targets")
FULL_STATE_BREAKING = ("optimiser", "schedule", "train.epochs", "train.samples_per_epoch", "train.batch_size")
START_DIR = "clone"


class EpochUnavailable(ValueError):
    """The state of the requested epoch can no longer be restored exactly."""


def stored_checkpoints(run_dir: Path) -> dict:
    """Epoch -> file of every stored epoch of a run (``epoch_NNNN.pt``, ``latest.pt``)."""
    checkpoints = Path(run_dir) / "checkpoints"
    stored = {int(path.stem.removeprefix("epoch_")): path for path in checkpoints.glob("epoch_*.pt")}
    if (checkpoints / "latest.pt").exists():
        stored[next_epoch(run_dir) - 1] = checkpoints / "latest.pt"
    return dict(sorted(stored.items()))


def start_checkpoint(run_dir: Path, from_epoch: int | None) -> tuple:
    """
    The stored checkpoint a clone starts from.

    Parameters
    ----------
    run_dir : Path
        Run directory cloned.
    from_epoch : int or None
        Stored epoch; None takes the latest.

    Returns
    -------
    tuple
        ``(epoch, path)``.

    Raises
    ------
    FileNotFoundError
        If the run has no checkpoint.
    EpochUnavailable
        If ``from_epoch`` is not stored, naming the stored epochs.
    """
    stored = stored_checkpoints(run_dir)
    if not stored:
        raise FileNotFoundError(f"{run_dir}: no checkpoint to clone from")
    epoch = max(stored) if from_epoch is None else from_epoch
    if epoch not in stored:
        raise EpochUnavailable(f"{run_dir}: epoch {epoch} is not stored; stored epochs: {', '.join(map(str, stored))}")
    return epoch, stored[epoch]


def state_mismatches(settings: dict, record: dict, full_state: bool) -> list:
    """
    Changes from a checkpoint's settings that its saved state cannot take.

    Parameters
    ----------
    settings : dict
        Resolved settings of the new run.
    record : dict
        The checkpoint's record (``run`` and ``derived`` as YAML text).
    full_state : bool
        Whether the optimiser and schedule states are restored too.

    Returns
    -------
    list of str
        One ``location: old -> new`` entry per change, window geometry included.
    """
    old = resolve_settings(load_run_record(record["run"]))
    found = []
    for location in BREAKING + (FULL_STATE_BREAKING if full_state else ()):
        keys = location.split(".")
        before, after = (functools.reduce(operator.getitem, keys, values) for values in (old, settings))
        found += differences({location: before}, {location: after})
    window = dataclasses.asdict(build_data(settings["data"])[0])
    return found + differences(yaml.safe_load(record["derived"])["window"], window, "window")


def best_up_to(run_dir: Path, epoch: int, best_loss: float) -> Path | None:
    """
    The stored checkpoint with validation loss ``best_loss`` at or before ``epoch``.

    Parameters
    ----------
    run_dir : Path
        Run directory cloned.
    epoch : int
        Epoch cloned.
    best_loss : float
        That epoch's ``best_validation_loss``.

    Returns
    -------
    Path or None
        None if it is no longer stored.
    """
    for path in [Path(run_dir) / "checkpoints" / "best.pt", *stored_checkpoints(run_dir).values()]:
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            if state["epoch"] <= epoch and state["validation_loss"] == best_loss:
                return path
    return None


def full_state_epochs(run_dir: Path) -> list:
    """Stored epochs whose best checkpoint up to them is stored too."""
    epochs = []
    for epoch, path in stored_checkpoints(run_dir).items():
        best_loss = torch.load(path, map_location="cpu", weights_only=False, mmap=True)["best_validation_loss"]
        if best_loss == float("inf") or best_up_to(run_dir, epoch, best_loss) is not None:
            epochs.append(epoch)
    return epochs


def prior_copy(old_dir: Path, new_dir: Path, setting: str) -> list:
    """``[(source, target)]`` for a prior file ``signal.prior`` names inside the run directory, else empty."""
    source, target = (Path(old_dir) / setting).resolve(), (Path(new_dir) / setting).resolve()
    return [(source, target)] if target.is_relative_to(Path(new_dir).resolve()) else []


def copy_history(old_dir: Path, new_dir: Path, epoch: int) -> None:
    """Copy ``results/losses.h5`` and ``validation.h5`` up to ``epoch``."""
    old, new = Path(old_dir) / "results", Path(new_dir) / "results"
    new.mkdir(parents=True, exist_ok=True)
    if (old / "losses.h5").exists():
        shutil.copy2(old / "losses.h5", new / "losses.h5")
        with h5py.File(new / "losses.h5", "a") as file:
            for split in ("training", "validation"):
                file[split]["loss"][epoch + 1:] = 0.0  # the rows' fill value
    if (old / "validation.h5").exists():
        with h5py.File(old / "validation.h5", "r") as source, h5py.File(new / "validation.h5", "w") as target:
            for name in source:
                if name.startswith("epoch_") and int(name.removeprefix("epoch_")) <= epoch:
                    source.copy(source[name], target, name)


def clone_run(old_dir: Path, new_dir: Path, settings: dict, full_state: bool, from_epoch: int | None) -> tuple:
    """
    Start ``new_dir`` from a stored checkpoint of ``old_dir``.

    Full state also copies the best checkpoint up to that epoch (to
    ``clone/best.pt``), the mining bank and the results up to that epoch.

    Parameters
    ----------
    old_dir, new_dir : Path
        Run directory cloned, and the new one (absent or empty).
    settings : dict
        Settings of ``old_dir/run.yaml``, from ``check_run``.
    full_state : bool
        Restore weights, optimiser, schedule, EMA and mining bank, continuing
        the epoch count; else the weights only, from epoch 0.
    from_epoch : int or None
        Stored epoch to start from; None takes the latest.

    Returns
    -------
    tuple
        ``(epoch, changes)``: the epoch cloned, and every setting that
        differs from its checkpoint as ``location: old -> new``.

    Raises
    ------
    FileExistsError
        If ``new_dir`` is not empty.
    FileNotFoundError
        If ``old_dir`` has no checkpoint.
    StateMismatch
        On a change the saved state cannot take.
    EpochUnavailable
        If ``from_epoch`` is not stored, or with full state, the best
        checkpoint up to it is no longer stored.
    """
    old_dir, new_dir = Path(old_dir), Path(new_dir)
    if new_dir.exists() and any(new_dir.iterdir()):
        raise FileExistsError(f"{new_dir} is not empty")
    epoch, path = start_checkpoint(old_dir, from_epoch)
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    record, new = state["record"], resolve_settings(settings)
    mismatches = state_mismatches(new, record, full_state)
    if mismatches:
        raise StateMismatch(f"{old_dir}/run.yaml differs from {path.name} (epoch {epoch}) in settings its saved state "
                            f"cannot take: {'; '.join(mismatches)}", not state_mismatches(new, record, False))
    start, prior = new_dir / START_DIR, settings["signal"]["prior"]
    copies = [(old_dir / "run.yaml", new_dir / "run.yaml"), (path, start / "start.pt")]
    copies += prior_copy(old_dir, new_dir, prior) if prior.endswith(".yaml") else []
    if full_state and state["best_validation_loss"] != float("inf"):
        best = best_up_to(old_dir, epoch, state["best_validation_loss"])
        if best is None:
            raise EpochUnavailable(f"{old_dir}: the best checkpoint up to epoch {epoch} is no longer stored; full state "
                                   f"can start from epochs {', '.join(map(str, full_state_epochs(old_dir)))}")
        copies.append((best, start / "best.pt"))
    if full_state:
        bank = old_dir / "mining" / "bank.h5"
        copies += [(bank, new_dir / "mining" / "bank.h5")] if bank.exists() else []

    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if full_state:
        copy_history(old_dir, new_dir, epoch)
    write_yaml(new_dir / "resolved" / "clone.yaml", {
        "source": str(old_dir.resolve()), "checkpoint": path.name, "epoch": epoch,
        "state": "full" if full_state else "weights",
    })
    return epoch, differences(resolve_settings(load_run_record(record["run"])), new)


def clone_start(run_dir: Path) -> tuple | None:
    """
    The checkpoint a cloned run starts from, until its first checkpoint is saved.

    Parameters
    ----------
    run_dir : Path
        Run directory.

    Returns
    -------
    tuple or None
        ``(state, full_state)``; None for a run not cloned or already started.
    """
    run_dir = Path(run_dir)
    if training_started(run_dir) or not (run_dir / START_DIR / "start.pt").exists():
        return None
    state = torch.load(run_dir / START_DIR / "start.pt", map_location="cpu", weights_only=False, mmap=True)
    return state, yaml.safe_load((run_dir / "resolved" / "clone.yaml").read_text())["state"] == "full"


def check_start(settings: dict, state: dict, full_state: bool) -> list:
    """
    Check a cloned run's settings against its start checkpoint.

    Parameters
    ----------
    settings : dict
        Resolved settings of the run.
    state : dict
        The start checkpoint.
    full_state : bool
        Whether the clone restores the full state.

    Returns
    -------
    list of str
        Every setting that differs from the checkpoint's.

    Raises
    ------
    StateMismatch
        On a change the saved state cannot take.
    """
    mismatches = state_mismatches(settings, state["record"], full_state)
    if mismatches:
        raise StateMismatch(f"run.yaml differs from the clone's start checkpoint (epoch {state['epoch']}) in settings "
                            f"its saved state cannot take: {'; '.join(mismatches)}",
                            not state_mismatches(settings, state["record"], False))
    return differences(resolve_settings(load_run_record(state["record"]["run"])), settings)


def check_built(derived: dict, state: dict, scaler, full_state: bool) -> None:
    """
    Check a built cloned run against its start checkpoint: multirate bins and, with full state, the loss scaler.

    Parameters
    ----------
    derived : dict
        The run's derived values.
    state : dict
        The start checkpoint.
    scaler : torch.amp.GradScaler or None
        The run's loss scaler.
    full_state : bool
        Whether the clone restores the full state.

    Raises
    ------
    StateMismatch
        If the bins differ, or the scaler's presence does.
    """
    if derived["multirate_bins"] != yaml.safe_load(state["record"]["derived"])["multirate_bins"]:
        raise StateMismatch("multirate bins differ from the clone's start checkpoint "
                            "(set by preprocess.multirate, the window and the prior's tc bounds)", False)
    if full_state and (scaler is None) != (state["scaler"] is None):
        precision = load_run_record(state["record"]["run"])["train"]["precision"]
        raise StateMismatch(f"train.precision differs from the clone's start checkpoint ({precision}) in its use of a "
                            "loss scaler, whose state full state restores", True)


def install_full_state(run_dir: Path, state: dict, ema: bool) -> None:
    """
    Make a full-state clone's start checkpoint the run's ``checkpoints/``, as if trained to that epoch.

    Parameters
    ----------
    run_dir : Path
        Run directory, not yet started.
    state : dict
        The start checkpoint.
    ema : bool
        Whether the run keeps an EMA (``ema.pt``, from the checkpoint's EMA or else its weights).
    """
    start, part = Path(run_dir) / START_DIR, Path(run_dir) / "checkpoints.part"
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir()
    shutil.copy2(start / "start.pt", part / "latest.pt")
    if (start / "best.pt").exists():
        shutil.copy2(start / "best.pt", part / "best.pt")
    if ema:
        averaged = state["model"] if state["ema"] is None else {
            key.removeprefix("module."): value for key, value in state["ema"].items() if key.startswith("module.")
        }
        torch.save({**state, **dict.fromkeys(TRAINING_ONLY), "model": averaged}, part / "ema.pt")
    (part / "progress.json").write_text(json.dumps({"epoch": state["epoch"]}))
    os.replace(part, Path(run_dir) / "checkpoints")
