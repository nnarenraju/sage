#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : checkpoint.py
Description     : Short description of the file

Created on 2026-03-22 11:10:40

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import os
import torch
import shutil
import json
import random
import numpy as np
from datetime import datetime
from functools import cached_property


def _atomic_torch_save(obj, path):
    """``torch.save`` to a sibling temp file then ``os.replace`` (atomic on the
    same filesystem), so a crash/pre-emption mid-write can never truncate an
    existing good checkpoint."""
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json_dump(obj, path):
    """``json.dump`` via a temp file + atomic ``os.replace`` (same rationale)."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def config_to_dict(cfg):
    """Flatten a Sage config into a plain, JSON-serialisable dict of values.

    ``BaseConfig`` / ``BaseDataConfig`` are ``__getattr__`` forwarders whose own
    instance dict is just ``{"cfg": <wrapped object>}``. Serialising
    ``cfg.__dict__`` therefore wrote a single useless entry::

        {"cfg": "<config_HL.O3bCFG object at 0x7f38...>"}

    and the run's actual hyperparameters -- learning rate schedule, epochs,
    detectors, everything -- were never recorded. That makes a finished run
    impossible to interrogate afterwards, which is exactly when you want to
    know what it was configured with.

    The real settings live as *class* attributes on the wrapped config object,
    so this walks the class as well as the instance, and also materialises the
    derived ``cached_property`` values (``padded_length_in_s``, ``delta_f``,
    ...) so the snapshot records what the run actually used rather than only
    what was typed.

    Parameters
    ----------
    cfg : object
        A ``BaseConfig``/``BaseDataConfig`` wrapper, or a raw config object.

    Returns
    -------
    dict
        Attribute name -> value, for every public, non-callable setting.
    """
    seen = {}

    def public_attrs(obj):
        out = {}
        # Class attributes: where a `class MyCFG: batch_size = 64` config keeps
        # its values. Walk the MRO so subclassed configs are covered too.
        for klass in reversed(type(obj).__mro__):
            if klass is object:
                continue
            for name, value in vars(klass).items():
                if name.startswith("_") or callable(value):
                    continue
                if isinstance(value, (property, staticmethod, classmethod)):
                    continue
                out[name] = value
        # Instance attributes take precedence.
        for name, value in vars(obj).items():
            if not name.startswith("_") and not callable(value):
                out[name] = value
        return out

    # Unwrap the forwarder, keeping anything set on the wrapper itself.
    raw = getattr(cfg, "cfg", None) or getattr(cfg, "data_cfg", None)
    if raw is not None and raw is not cfg:
        seen.update(public_attrs(raw))
    seen.update({k: v for k, v in public_attrs(cfg).items()
                 if k not in ("cfg", "data_cfg")})

    # Derived cached_property values (delta_f, padded_length_in_s, ...): these
    # are what the pipeline actually consumed, so record them too.
    for klass in type(cfg).__mro__:
        for name, value in vars(klass).items():
            if name.startswith("_") or not isinstance(value, cached_property):
                continue
            try:
                seen[name] = getattr(cfg, name)
            except Exception:
                pass

    return seen


class CheckpointManager:
    """
    Manages saving and loading of training checkpoints.

    On construction, creates the checkpoint directory and writes one-time
    JSON snapshots of ``cfg`` and ``data_cfg`` so that the hyperparameters
    that produced each checkpoint are always recoverable.

    Three checkpoint types are maintained:

    * ``latest.pt`` — overwritten every call to :meth:`save`; safe resume
      point after a crash or pre-emption.
    * ``epoch_{N}.pt`` — per-epoch copy (optional; controlled by
      ``save_epoch_ckpt``).  Allows rolling back to any specific epoch.
    * ``best.pt`` — copy of ``latest.pt`` whenever ``val_loss`` improves;
      the recommended checkpoint for inference.

    Each checkpoint file contains the full training state:
    model weights, optimiser and scheduler states, AMP scaler state,
    configurations, and all four RNG states (Python, NumPy, CPU torch,
    CUDA torch) for bit-exact reproducibility.

    Parameters
    ----------
    cfg : BaseConfig
        Training configuration (provides ``export_dir``, ``autocast``,
        ``dtype``).
    data_cfg : BaseDataConfig
        Data configuration (saved to JSON snapshot only).
    model : nn.Module
        The model being trained (may be a ``torch.compile`` wrapper).
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler._LRScheduler
    scaler : torch.amp.GradScaler
    """

    def __init__(
        self,
        cfg,
        data_cfg,
        model,
        optimizer,
        scheduler,
        scaler,
    ):

        self.cfg = cfg
        self.data_cfg = data_cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

        self.best_val_loss = float("inf")

        self.ckpt_dir = os.path.join(cfg.export_dir, "CHECKPOINTS")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.latest_path = os.path.join(self.ckpt_dir, "latest.pt")
        self.best_path = os.path.join(self.ckpt_dir, "best.pt")

        # ---- save config snapshots once ----
        with open(os.path.join(self.ckpt_dir, "cfg_snapshot.json"), "w") as f:
            json.dump(config_to_dict(cfg), f, indent=2, default=str, sort_keys=True)

        with open(os.path.join(self.ckpt_dir, "data_cfg_snapshot.json"), "w") as f:
            json.dump(config_to_dict(data_cfg), f, indent=2, default=str, sort_keys=True)

    # ============================================================
    # STATE GATHER
    # ============================================================

    def _gather_state(self, epoch, val_loss=None):
        """
        Collect all serialisable training state into a flat dictionary.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).
        val_loss : float or None
            Validation loss at this epoch (stored for reference; ``None``
            if validation was not run this epoch).

        Returns
        -------
        dict
            All state needed to fully resume training, including model
            weights, optimiser/scheduler/scaler states, configs, parameter
            counts, and all four RNG states.
        """

        state = {
            # ---- bookkeeping ----
            "epoch": epoch,
            "timestamp": str(datetime.now()),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            # ---- training objects ----
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            # ---- configs ----
            # Store a flattened, plain-dict snapshot (identical to
            # cfg_snapshot.json) rather than the live config object. The raw
            # object pickles to a useless "<... object at 0x...>" anyway (see
            # config_to_dict) and its class may be unpicklable -- e.g. a config
            # built by a factory (make_configs.<locals>.O3aCFG). Nothing on
            # resume reads these keys; they are purely a record of settings.
            "cfg": json.loads(json.dumps(config_to_dict(self.cfg), default=str)),
            "data_cfg": json.loads(
                json.dumps(config_to_dict(self.data_cfg), default=str)),
            # ---- model info ----
            "param_count": sum(p.numel() for p in self.model.parameters()),
            "trainable_param_count": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            # ---- RNG states (CRITICAL FOR REPRO) ----
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            # ---- AMP metadata ----
            "amp_enabled": self.cfg.autocast,
            "amp_dtype": str(self.cfg.dtype),
        }

        return state

    # ============================================================
    # SAVE
    # ============================================================

    def save(self, epoch, val_loss=None, save_epoch_ckpt=True):
        """
        Save the current training state to disk.

        Always writes ``latest.pt``.  If ``val_loss`` is better than the
        current best, also copies to ``best.pt``.

        Parameters
        ----------
        epoch : int
            Current epoch (0-based).
        val_loss : float or None
            Validation loss; used to decide whether to update ``best.pt``.
        save_epoch_ckpt : bool
            If ``True`` (default), also write ``epoch_{epoch}.pt``.
        """

        # Decide "best" BEFORE gathering state so best_val_loss is persisted at
        # its current value and survives a resume. (Previously best_val_loss reset
        # to inf on every restart, so best.pt was clobbered at each 2-day segment
        # boundary; and it tracked whatever loss was passed in.)
        improved = val_loss is not None and val_loss < self.best_val_loss
        if improved:
            self.best_val_loss = val_loss

        state = self._gather_state(epoch, val_loss)

        # ---- latest (atomic: temp file + os.replace, so a kill mid-write leaves
        #      the previous good latest.pt intact instead of truncating it) ------
        _atomic_torch_save(state, self.latest_path)

        # Lightweight resume marker: lets a restarting process read the resume
        # epoch (to seed the data samplers resume-aware) WITHOUT loading the full
        # checkpoint. Written (atomically) after latest.pt so it never points past
        # an incomplete latest.pt.
        _atomic_json_dump(
            {"epoch": epoch, "val_loss": val_loss},
            os.path.join(self.ckpt_dir, "progress.json"),
        )

        # ---- epoch history ----
        if save_epoch_ckpt:
            _atomic_torch_save(
                state, os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt")
            )
            # Optional retention: keep only the newest ``keep_last_ckpts`` epoch
            # copies. Default None => keep everything (nothing is ever deleted
            # unless a run opts in via cfg.keep_last_ckpts). latest.pt/best.pt are
            # always kept.
            self._prune_epoch_ckpts(getattr(self.cfg, "keep_last_ckpts", None))

        # ---- best (atomic write of THIS state; only when val_loss improved) ----
        if improved:
            print(f"New BEST checkpoint at epoch {epoch} | val_loss={val_loss:.6f}")
            _atomic_torch_save(state, self.best_path)

    def _prune_epoch_ckpts(self, keep_last):
        """Delete all but the newest ``keep_last`` ``epoch_{N}.pt`` files.
        ``None`` or ``<= 0`` keeps everything (no deletion). ``latest.pt`` and
        ``best.pt`` are never touched."""
        if keep_last is None or keep_last <= 0:
            return
        import glob, re
        def _epn(p):
            m = re.search(r"epoch_(\d+)\.pt$", p)
            return int(m.group(1)) if m else -1
        files = sorted(
            glob.glob(os.path.join(self.ckpt_dir, "epoch_*.pt")), key=_epn
        )
        for p in files[:-keep_last]:
            try:
                os.remove(p)
            except OSError:
                pass

    def _load_newest_epoch_ckpt(self, map_location):
        """Load the newest intact ``epoch_{N}.pt`` (resume fallback when
        ``latest.pt`` is unreadable, e.g. truncated by a pre-atomic write)."""
        import glob, re
        def _epn(p):
            m = re.search(r"epoch_(\d+)\.pt$", p)
            return int(m.group(1)) if m else -1
        files = sorted(
            glob.glob(os.path.join(self.ckpt_dir, "epoch_*.pt")),
            key=_epn, reverse=True,
        )
        for p in files:
            try:
                return torch.load(p, map_location=map_location, weights_only=False)
            except Exception:
                continue
        raise RuntimeError(
            "No loadable checkpoint (latest.pt and all epoch_*.pt failed)"
        )

    # ============================================================
    # LOAD
    # ============================================================

    def load_latest(self, map_location="cpu"):
        """Load latest snapshot and resume training

        Args:
            map_location (str, optional): _description_. Defaults to "cpu".

        Returns:
            _type_: _description_

        Usage:
            start_epoch = ckpt_mgr.load_latest(map_location=cfg.device)
            for nepoch in range(start_epoch, cfg.num_epochs):
            ...
        """
        print("Loading latest checkpoint")
        # weights_only=False: our checkpoints intentionally carry non-tensor
        # state (cfg/data_cfg objects + numpy/python RNG) and are self-produced
        # and trusted; torch>=2.6 defaults weights_only=True which rejects them.
        try:
            ckpt = torch.load(self.latest_path, map_location=map_location,
                              weights_only=False)
        except Exception as e:
            # latest.pt unreadable (e.g. truncated by a kill mid-write on an old
            # pre-atomic checkpoint). Fall back to the newest intact epoch_*.pt
            # so the automated resume chain recovers instead of crash-looping.
            print(f"[resume] latest.pt failed to load ({e}); "
                  f"falling back to newest epoch checkpoint")
            ckpt = self._load_newest_epoch_ckpt(map_location)
        self._restore(ckpt)
        return ckpt["epoch"] + 1

    @staticmethod
    def peek_next_epoch(export_dir, map_location="cpu"):
        """Resume epoch (last saved epoch + 1), read from the lightweight
        ``progress.json`` marker WITHOUT loading the full checkpoint. Returns
        ``0`` when no checkpoint exists yet.

        This lets a restarting run seed its data samplers with a resume-aware
        seed *before* they are built, so the resumed stream never replays the
        pre-crash draws. Mirrors :meth:`load_latest`'s ``epoch + 1``.
        """
        ckpt_dir = os.path.join(export_dir, "CHECKPOINTS")
        marker = os.path.join(ckpt_dir, "progress.json")
        if os.path.exists(marker):
            with open(marker) as f:
                return int(json.load(f)["epoch"]) + 1
        # Fallback for checkpoints written before progress.json existed: read the
        # epoch out of latest.pt (heavier, but only hit on legacy resumes).
        latest = os.path.join(ckpt_dir, "latest.pt")
        if os.path.exists(latest):
            ckpt = torch.load(latest, map_location=map_location, weights_only=False)
            return int(ckpt["epoch"]) + 1
        return 0

    def load_best(self, map_location="cpu"):
        """Load best snapshot for inference

        Args:
            map_location (str, optional): _description_. Defaults to "cpu".

        Returns:
            _type_: _description_

        Usage:
            ckpt_mgr.load_best(map_location=cfg.device)
            model.eval()
        """
        print("Loading best checkpoint")
        ckpt = torch.load(self.best_path, map_location=map_location,
                          weights_only=False)
        self._restore(ckpt)
        return ckpt

    def load_epoch(self, epoch, map_location="cpu"):
        """
        Load a specific per-epoch checkpoint.

        Parameters
        ----------
        epoch : int
            Epoch index of the checkpoint to load.
        map_location : str
            Device mapping for ``torch.load``.

        Returns
        -------
        int
            ``epoch + 1`` — the next epoch to run.
        """
        path = os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt")
        print(f"Loading checkpoint epoch {epoch}")
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        self._restore(ckpt)
        return ckpt["epoch"] + 1

    # ============================================================
    # RESTORE STATE
    # ============================================================

    def _restore(self, ckpt):
        """
        Apply a loaded checkpoint dict to the current training objects.

        Restores model weights, optimiser/scheduler/scaler states, and all
        four RNG states (Python, NumPy, CPU torch, CUDA torch).

        Parameters
        ----------
        ckpt : dict
            Dictionary as produced by :meth:`_gather_state`.
        """

        # Restore the best-val-loss marker so best.pt tracking survives a resume
        # (absent on pre-fix checkpoints -> fall back to +inf).
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        # ---- RNG restore ----
        # torch.load(map_location=<cuda>) moves EVERY checkpoint tensor to the
        # device, including the RNG-state ByteTensors -- but set_rng_state()
        # requires them on CPU (else "RNG state must be a torch.ByteTensor").
        # Force them back to CPU. (CPU-only resume never hit this.)
        torch.set_rng_state(ckpt["torch_rng"].cpu())

        if torch.cuda.is_available() and ckpt["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng"]])

        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])
