#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : conftest.py
Description   : Shared fixtures and the environment gates used across the suite.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Most of the suite runs anywhere. The parts that do not are gated by probing for the thing
they need rather than by an environment variable, so a machine that happens to have the
data runs those tests without being told to, and a machine that does not says which probe
failed instead of reporting a bare skip.
"""

import os
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

# Nothing this project runs may write to /tmp, and pytest's tmp_path lands there by
# default. Redirected before any fixture asks for a temporary directory, since the temp
# root is resolved lazily on first use. Set here rather than in addopts so it can still be
# overridden per machine, and so a stale basetemp is not wiped from under a parallel run.
_TEMPROOT = Path(
    os.environ.get("SAGE_PYTEST_TEMPROOT", Path.home() / ".cache" / "sage-pytest")
)
_TEMPROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(_TEMPROOT))

# Project storage. Resolved from the server registry when it is importable, since that is
# the single source of truth for where releases live, and falls back to the configured
# root so the probes still work in a bare checkout.
WORK_ROOT = Path(os.environ.get("SAGE_WORK_ROOT", "/work/nagarajan"))

O3A_RELEASE = WORK_ROOT / "data_release_o3a"
O3B_RELEASE = WORK_ROOT / "data_release"
FIDUCIAL_O3AB = WORK_ROOT / "sage_runs" / "fiducial_psds_o3ab"
SEARCH_CHECKPOINT = (
    WORK_ROOT / "sage_runs" / "o3b" / "production_run_HL" / "CHECKPOINTS" / "best.pt"
)

_HAS_CUDA = torch.cuda.is_available()
_HAS_O3A = (O3A_RELEASE / "data_H1_O3a.bin").exists()
_HAS_O3B = (O3B_RELEASE / "data_H1_O3b.bin").exists()
_HAS_FIDUCIAL = (FIDUCIAL_O3AB / "fiducial_H1_psd.bin").exists()
_HAS_CHECKPOINT = SEARCH_CHECKPOINT.exists()
_HAS_REFERENCES = (REPO_ROOT / "docs" / "references").is_dir()

requires_gpu = pytest.mark.skipif(not _HAS_CUDA, reason="needs a CUDA device")

requires_o3a = pytest.mark.skipif(
    not (_HAS_O3A and _HAS_FIDUCIAL),
    reason=f"needs the O3a release ({_HAS_O3A}) and fiducial spectra ({_HAS_FIDUCIAL})",
)

requires_o3b = pytest.mark.skipif(
    not (_HAS_O3B and _HAS_FIDUCIAL),
    reason=f"needs the O3b release ({_HAS_O3B}) and fiducial spectra ({_HAS_FIDUCIAL})",
)

requires_checkpoint = pytest.mark.skipif(
    not _HAS_CHECKPOINT, reason=f"needs a trained checkpoint at {SEARCH_CHECKPOINT}"
)

requires_search_env = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_O3A and _HAS_FIDUCIAL and _HAS_CHECKPOINT),
    reason=(
        f"needs CUDA ({_HAS_CUDA}), the O3a release ({_HAS_O3A}), fiducial spectra "
        f"({_HAS_FIDUCIAL}) and a checkpoint ({_HAS_CHECKPOINT})"
    ),
)

requires_references = pytest.mark.skipif(
    not _HAS_REFERENCES,
    reason="needs docs/references; run docs/references/fetch.py to populate it",
)


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def float_dtype():
    return torch.float32


@pytest.fixture(scope="session")
def repo_root():
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture
def synthetic_release(tmp_path):
    """A miniature two-detector strain release with the real awkward properties."""
    from tests.search_fixtures import make_synthetic_release

    return make_synthetic_release(tmp_path / "release", detectors=("H1", "L1"))


@pytest.fixture
def synthetic_release_hlv(tmp_path):
    """The same, for a three-detector network."""
    from tests.search_fixtures import make_synthetic_release

    return make_synthetic_release(
        tmp_path / "release_hlv", detectors=("H1", "L1", "V1")
    )


@pytest.fixture
def synthetic_checkpoint(tmp_path):
    """A checkpoint in the current flat-config format, with a separable frontend."""
    from tests.search_fixtures import make_synthetic_checkpoint

    return make_synthetic_checkpoint(tmp_path / "CHECKPOINTS" / "best.pt")
