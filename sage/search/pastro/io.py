#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : io.py
Description   : Products and contract enforcement for the p_astro stage.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The background used for the noise density is recorded with the products and asserted to
match the one behind the false-alarm rates. If the two differed, a candidate's
significance and its probability would rest on different noise models and could disagree
precisely where it matters most.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


class ContractViolation(ValueError):
    """Raised when an input breaks a documented precondition of the stage."""


def require_clustered(table) -> None:
    """Refuse a trigger set that has not been clustered."""
    raise NotImplementedError


def require_matching_background(pastro_attrs: Dict[str, object], far_attrs: Dict[str, object]) -> None:
    """Assert the noise density and the false-alarm rates share a background."""
    raise NotImplementedError


def save_model(
    path: str | Path,
    densities: Dict[str, object],
    support,
    posterior,
    validation,
    attrs: Optional[Dict[str, object]] = None,
) -> Path:
    """Write the fitted model, its support and its validation record."""
    raise NotImplementedError


def load_model(path: str | Path):
    """Read a persisted model."""
    raise NotImplementedError
