#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : manifest.py
Description   : Provenance attrs, run manifest and the stage journal.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Every product carries enough provenance to be re-derived: code version, spec hash,
checkpoint identity, seeds, and the exact livetimes behind any rate.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROVENANCE_KEYS = (
    "schema_version",
    "sage_version",
    "git_hash",
    "git_dirty",
    "spec_hash",
    "config_module",
    "checkpoint_path",
    "checkpoint_sha256",
    "observing_run",
    "detectors",
    "sample_rate",
    "window_samples",
    "stride_samples",
    "seed",
    "created_utc",
)


def provenance(spec, ckpt=None, **extra) -> Dict[str, Any]:
    """Build the provenance attr block written onto every output file."""
    raise NotImplementedError


def stamp(handle, attrs: Dict[str, Any]) -> None:
    """Attach a provenance block to an open HDF5 handle."""
    raise NotImplementedError


def verify(path: str | Path, expect_spec_hash: Optional[str] = None) -> Dict[str, Any]:
    """Read a product's provenance and optionally assert the spec hash."""
    raise NotImplementedError


@dataclass
class RunManifest:
    """Campaign-level summary: livetimes, throughput, stage completion."""

    path: Path

    def record_stage(self, stage: str, report: Dict[str, Any]) -> None:
        """Append a completed stage and its report."""
        raise NotImplementedError

    def record_livetime(self, run: str, coverage: Dict[str, Any]) -> None:
        """Store the livetime decomposition for one observing run."""
        raise NotImplementedError

    def summary(self) -> Dict[str, Any]:
        """Everything needed for the methods section of the paper."""
        raise NotImplementedError


def journal(path: str | Path, event: Dict[str, Any]) -> None:
    """Append one line to the append-only stage journal."""
    raise NotImplementedError
