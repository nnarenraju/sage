"""Training checkpoints, in one self-contained format.

Every file holds the run record and the training state, so a checkpoint alone
is enough to resume, evaluate or search::

    record                 resolved run.yaml, prior.yaml, derived.yaml and
                           versions.yaml as text; the fiducial files (name -> bytes)
    epoch                  last finished epoch
    validation_loss        this epoch's, or None
    best_validation_loss
    model                  weights (uncompiled names)
    optimiser, schedule    states
    scaler, ema            states, or None without a loss scaler or EMA
    rng                    torch, cuda, numpy and python generator states
    timestamp

``checkpoints/`` holds ``latest.pt`` (every epoch), ``best.pt`` (lowest
validation loss), ``epoch_NNNN.pt`` (the newest ``keep`` epochs) and, with EMA,
``ema.pt``: the averaged weights as ``model``, training-only entries None; after
training, ``ema_calibrated.pt`` is ``ema.pt`` with recalibrated batch-norm
statistics.
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

RECORD_FILES = ("run", "prior", "derived", "versions")
TRAINING_ONLY = ("optimiser", "schedule", "scaler", "ema", "rng")


def atomic_torch_save(obj, path):
    """``torch.save`` to a sibling temp file then ``os.replace`` (atomic on the
    same filesystem), so a crash mid-write never truncates a good checkpoint."""
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_dump(obj, path):
    """``json.dump`` via a temp file + atomic ``os.replace`` (same rationale)."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def rng_state() -> dict:
    """States of the global torch, CUDA, NumPy and Python generators."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def set_rng_state(state: dict) -> None:
    """Restore the global generators from :func:`rng_state`."""
    # map_location moves the generator states too; set_rng_state needs them on CPU.
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def restore(state: dict, model, optimiser, schedule, scaler, ema_model) -> None:
    """
    Load a checkpoint's training state into a run and the global generators.

    Parameters
    ----------
    state : dict
        A checkpoint.
    model, optimiser, schedule, scaler, ema_model
        As :class:`CheckpointManager` takes them.
    """
    model.load_state_dict(state["model"])
    optimiser.load_state_dict(state["optimiser"])
    schedule.load_state_dict(state["schedule"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    if ema_model is not None and state["ema"] is not None:  # a clone turning EMA on keeps it at the cloned weights
        ema_model.load_state_dict(state["ema"])
    set_rng_state(state["rng"])


def run_record(run_dir: Path) -> dict:
    """
    The record every checkpoint of a run carries.

    Parameters
    ----------
    run_dir : Path
        Run directory, its ``resolved/`` written.

    Returns
    -------
    dict
        ``run``, ``prior``, ``derived``, ``versions`` (YAML text) and
        ``fiducial`` (file name to bytes).
    """
    resolved = Path(run_dir) / "resolved"
    record = {name: (resolved / f"{name}.yaml").read_text() for name in RECORD_FILES}
    record["fiducial"] = {path.name: path.read_bytes() for path in sorted((resolved / "fiducial").iterdir())}
    return record


def next_epoch(run_dir: Path) -> int:
    """The epoch a run starts at: last saved epoch + 1, or 0 before any checkpoint."""
    marker = Path(run_dir) / "checkpoints" / "progress.json"
    return json.loads(marker.read_text())["epoch"] + 1 if marker.exists() else 0


class CheckpointManager:
    """
    Saves a run's checkpoints every epoch and restores the latest on resume.

    Parameters
    ----------
    run_dir : Path
        Run directory; checkpoints go to ``checkpoints/``.
    model : nn.Module
        The network, uncompiled (so weight names carry no compile prefix).
    optimiser : torch.optim.Optimizer
    schedule : torch.optim.lr_scheduler.LRScheduler
    scaler : torch.amp.GradScaler or None
    ema_model : torch.optim.swa_utils.AveragedModel or None
        The EMA of the weights, when EMA is on.
    keep : int or str
        Epoch copies kept (the newest), or ``"all"``.
    """

    def __init__(self, run_dir, model, optimiser, schedule, scaler, ema_model, keep):
        self.dir = Path(run_dir) / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.record = run_record(run_dir)
        self.model = model
        self.optimiser = optimiser
        self.schedule = schedule
        self.scaler = scaler
        self.ema_model = ema_model
        self.keep = keep
        self.best_validation_loss = float("inf")

    def _state(self, epoch, validation_loss):
        return {
            "record": self.record,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "best_validation_loss": self.best_validation_loss,
            "model": self.model.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "schedule": self.schedule.state_dict(),
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "ema": None if self.ema_model is None else self.ema_model.state_dict(),
            "rng": rng_state(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def save(self, epoch: int, validation_loss) -> None:
        """
        Write ``latest.pt``, ``epoch_NNNN.pt``, ``best.pt`` when the
        validation loss improved, and ``ema.pt`` with EMA.

        Parameters
        ----------
        epoch : int
            Epoch just finished.
        validation_loss : float or None
            This epoch's validation loss; None when validation did not run.
        """
        improved = validation_loss is not None and validation_loss < self.best_validation_loss
        if improved:
            self.best_validation_loss = validation_loss
        state = self._state(epoch, validation_loss)

        atomic_torch_save(state, self.dir / "latest.pt")
        atomic_json_dump({"epoch": epoch}, self.dir / "progress.json")
        atomic_torch_save(state, self.dir / f"epoch_{epoch:04d}.pt")
        if self.keep != "all":
            for path in sorted(self.dir.glob("epoch_*.pt"))[: -self.keep]:
                path.unlink()
        if improved:
            atomic_torch_save(state, self.dir / "best.pt")
        if self.ema_model is not None:
            ema = {**state, **dict.fromkeys(TRAINING_ONLY), "model": self.ema_model.module.state_dict()}
            atomic_torch_save(ema, self.dir / "ema.pt")

    def load_latest(self, map_location) -> int:
        """
        Restore the training state of ``latest.pt``.

        Parameters
        ----------
        map_location : str or torch.device
            Device the tensors are loaded onto.

        Returns
        -------
        int
            The epoch to continue at.
        """
        state = torch.load(self.dir / "latest.pt", map_location=map_location, weights_only=False)
        self.best_validation_loss = state["best_validation_loss"]
        restore(state, self.model, self.optimiser, self.schedule, self.scaler, self.ema_model)
        return state["epoch"] + 1
