#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : catalogue_json.py
Description   : Machine-readable catalogue for public distribution.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Published in the schema the open-data portal accepts, so the catalogue can be queried
through the same interface as every other event list rather than only as a table in a
paper.

Eligibility conditions are checked before anything is written: the analysis must cover
times where no event was previously known, the values must match the publication, and
the entries must meet the portal's inclusion threshold.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence


def eligibility(spec, candidates) -> Dict[str, object]:
    """Check the conditions for listing a catalogue publicly."""
    raise NotImplementedError


def to_json(
    candidates,
    spec,
    path: str | Path,
    reference: str = "",
    threshold_p_astro: float = 0.5,
) -> Path:
    """Write the catalogue in the portal's schema."""
    raise NotImplementedError


def validate(path: str | Path, schema: Optional[str | Path] = None) -> Dict[str, object]:
    """Validate the written file against the schema, offline."""
    raise NotImplementedError
