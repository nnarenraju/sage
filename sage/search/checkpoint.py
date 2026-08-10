#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : checkpoint.py
Description   : The checkpoint loader and stored-vs-live geometry validation.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Consolidates the three ad-hoc loaders in runs/o3b (eval_efficiency_snr.py,
benchmark_mlgwsc1.py, validate_checkpoint.py), which each re-implemented the
torch.compile prefix strip and none of which validated the stored config.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GEOMETRY_KEYS: Tuple[str, ...] = (
    "sample_rate",
    "sample_length_in_s",
    "padding_length_in_s",
    "detectors",
    "norm_type",
    "do_point_estimate",
    "noise_low_frequency_cutoff",
    "signal_low_frequency_cutoff",
)


@dataclass
class LoadedCheckpoint:
    """A checkpoint plus the flattened configs it was trained under."""

    path: Path
    sha256: str
    state_dict: Dict[str, Any]
    cfg: Dict[str, Any]
    data_cfg: Dict[str, Any]
    epoch: int
    val_loss: float

    @property
    def norm_type(self) -> str:
        """Normalisation layer the weights were trained with."""
        raise NotImplementedError

    @property
    def detectors(self) -> Tuple[str, ...]:
        """Detector ordering baked into the weights."""
        raise NotImplementedError

    def tc_prior(self) -> Tuple[float, float]:
        """Coalescence-time prior bounds recorded at training time."""
        raise NotImplementedError


def read_checkpoint(path: str | Path, map_location: str = "cpu") -> LoadedCheckpoint:
    """Load a ``.pt``, strip the ``_orig_mod.`` compile prefix and hash the file."""
    raise NotImplementedError


def validate_geometry(
    ckpt: LoadedCheckpoint,
    cfg,
    data_cfg,
    keys: Tuple[str, ...] = GEOMETRY_KEYS,
    strict: bool = True,
) -> List[str]:
    """
    Compare the stored config against the live one.

    Returns the list of mismatches; raises on any when ``strict``. The list is
    recorded in output provenance either way.
    """
    raise NotImplementedError


def build_search_model(ckpt: LoadedCheckpoint, device: str, dtype: str = "float32"):
    """Instantiate the architecture from the stored config and load the weights."""
    raise NotImplementedError


def assert_separable(model, sample_input=None) -> None:
    """
    Prove the per-detector path is separable, numerically.

    Perturbs one detector's samples and asserts every other detector's frontend
    output is bitwise unchanged. Required before the frontend cache may be used:
    InstanceNorm1d normalises per channel and is separable, whereas GroupNorm(1, D)
    spans the detector axis and is not. A config-string check does not survive a
    refactor, so this walks the module graph and then measures.
    """
    raise NotImplementedError


def load_search_model(
    path: str | Path,
    cfg,
    data_cfg,
    device: str = "cuda",
    strict: bool = True,
    require_separable: bool = False,
) -> Tuple[Any, LoadedCheckpoint]:
    """Read, validate, build and (optionally) prove separability in one call."""
    raise NotImplementedError
