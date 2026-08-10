#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : campaign.py
Description   : Resumable scoring of the injected stream.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Injection streams are scored one at a time so that the spacing assumption behind the
overlay association window holds.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclass
class CampaignReport:
    """Completion accounting for one injection pass."""

    stream: int
    n_injections: int
    n_scored: int
    n_outside_segments: int
    wall_seconds: float

    def as_dict(self) -> dict:
        """Flat dict for the manifest."""
        raise NotImplementedError


class InjectionCampaign:
    """Score one injection stream over one observing run."""

    def __init__(self, spec, engine, injections, plan, writer) -> None:
        raise NotImplementedError

    def run(self, resume: bool = True) -> CampaignReport:
        """Score every block, skipping those already complete."""
        raise NotImplementedError


def run_campaign(spec, stream: int = 0) -> CampaignReport:
    """Stage driver for one injection stream."""
    raise NotImplementedError
