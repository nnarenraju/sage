#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : cache.py
Description   : On-disk cache for remote catalogue and posterior data.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Remote sources are fetched once into a content-addressed cache and frozen with a
manifest, so an analysis is reproducible and does not depend on a live service. The
cache lives on project storage; nothing is written to the system temporary directory.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class CacheEntry:
    """One cached artefact."""

    url: str
    path: Path
    sha256: str
    retrieved_utc: str
    bytes: int


class CatalogueCache:
    """Content-addressed cache with a freeze manifest."""

    def __init__(self, root: str | Path) -> None:
        raise NotImplementedError

    def fetch(self, url: str, refresh: bool = False) -> CacheEntry:
        """Return a cached artefact, downloading it if absent."""
        raise NotImplementedError

    def freeze(self, path: str | Path) -> Path:
        """Write the manifest pinning every entry used by an analysis."""
        raise NotImplementedError

    def verify(self, manifest: str | Path) -> Dict[str, bool]:
        """Check the cache against a frozen manifest."""
        raise NotImplementedError

    def offline(self) -> bool:
        """Whether every entry in the manifest is present locally."""
        raise NotImplementedError
