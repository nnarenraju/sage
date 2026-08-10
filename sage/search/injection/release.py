#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : release.py
Description   : Registry and staging of published injection releases.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Releases are staged and checksummed once, then read locally; no analysis stage reaches
the network. Record identifiers are pinned because a release can gain new versions with
different contents under the same concept identifier.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class InjectionRelease:
    """A pinned, checksummed injection data release."""

    key: str
    record_id: str
    filename: str
    sha256: str
    observing_runs: Tuple[str, ...]
    description: str = ""

    @property
    def is_combined(self) -> bool:
        """Whether the release spans more than one observing run."""
        raise NotImplementedError


RELEASES: Dict[str, InjectionRelease] = {}


def register(release: InjectionRelease) -> None:
    """Add a release to the registry."""
    raise NotImplementedError


def stage(key: str, dest_dir: str | Path, verify: bool = True) -> Path:
    """Download (if absent) and verify a release under a local cache on /work."""
    raise NotImplementedError

def resolve(key: str) -> InjectionRelease:
    """Look up a registered release, raising with suggestions on a typo."""
    raise NotImplementedError
