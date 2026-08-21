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


#: Name of the manifest inside a cache root.
MANIFEST_NAME: str = "catalogue_cache.json"


class CatalogueCache:
    """
    Content-addressed cache with a freeze manifest.

    A catalogue comparison must be reproducible, and a live service is not: GWOSC adds
    events, revises significances and re-releases catalogues, so the same analysis run
    twice against the network gives two answers with nothing to say which was which. Each
    artefact is fetched once, stored under the SHA-256 of its bytes, and recorded in a
    manifest that pins the URL to that digest.

    Content-addressed rather than URL-addressed: two URLs sometimes serve the same
    catalogue, and one URL serves different bytes over time. The digest is what identifies
    the artefact; the URL is how it was obtained.
    """

    def __init__(self, root: str | Path, offline_only: bool = False) -> None:
        self.root = Path(root)
        self.offline_only = bool(offline_only)
        self.root.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, CacheEntry] = {}
        manifest = self.root / MANIFEST_NAME
        if manifest.is_file():
            self._entries = _read_manifest(manifest, self.root)

    def path_for(self, digest: str) -> Path:
        """Where an artefact with this digest lives."""
        return self.root / digest[:2] / digest

    def fetch(self, url: str, refresh: bool = False) -> CacheEntry:
        """
        Return a cached artefact, downloading it if absent.

        A cached entry is returned without touching the network, which is what makes a
        frozen analysis runnable offline and on a compute node with no route out.
        ``refresh`` forces a re-fetch, and is how a catalogue is deliberately updated --
        never a side effect of running the analysis again.
        """
        if not refresh and url in self._entries:
            entry = self._entries[url]
            if entry.path.is_file():
                return entry
        if self.offline_only:
            raise LookupError(
                f"{url} is not in the cache at {self.root} and this cache is offline. "
                "Populate it once with put() or a run that is allowed to fetch, then "
                "freeze it; an analysis that silently reaches the network is one whose "
                "inputs can change under it between runs"
            )
        return self._store(url, _download(url))

    def put(self, url: str, payload: bytes) -> CacheEntry:
        """
        Insert an artefact obtained elsewhere.

        For a file downloaded by hand, or copied from another machine: the cache does not
        care how bytes arrived, only that what it holds is identified by its digest.
        """
        return self._store(url, payload)

    def _store(self, url: str, payload: bytes) -> CacheEntry:
        """Write the bytes under their digest and record the entry."""
        import hashlib
        from datetime import datetime, timezone

        digest = hashlib.sha256(payload).hexdigest()
        target = self.path_for(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            # Written through a temporary in the same directory and renamed, so a kill
            # mid-write cannot leave a short file sitting under the digest of a whole one.
            temporary = target.with_suffix(".partial")
            temporary.write_bytes(payload)
            temporary.replace(target)
        entry = CacheEntry(
            url=str(url),
            path=target,
            sha256=digest,
            retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            bytes=len(payload),
        )
        self._entries[str(url)] = entry
        # Persisted on every store, not only at freeze(). The entry table is what makes a
        # later process reuse the artefact instead of re-fetching it, and a cache whose
        # index lives only in memory is not a cache -- the first run populates it, the
        # next one silently goes to the network, and "fetched once" is untrue.
        self.freeze(self.root / MANIFEST_NAME)
        return entry

    def freeze(self, path: str | Path) -> Path:
        """
        Write the manifest pinning every entry used by an analysis.

        The manifest is the reproducibility record: it names each URL, the digest that
        was read, and when. A campaign carrying one can be re-run years later against the
        catalogues it actually used rather than against whatever those URLs serve then.
        """
        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "root": str(self.root),
                    "entries": [
                        {
                            "url": entry.url,
                            "sha256": entry.sha256,
                            "retrieved_utc": entry.retrieved_utc,
                            "bytes": entry.bytes,
                        }
                        for entry in sorted(
                            self._entries.values(), key=lambda e: e.url
                        )
                    ],
                },
                indent=2,
            )
        )
        return target

    def verify(self, manifest: str | Path) -> Dict[str, bool]:
        """
        Check the cache against a frozen manifest.

        Each artefact is re-hashed rather than checked by size or timestamp. A truncated
        or partially-rewritten file keeps a plausible size, and the failure it produces --
        a catalogue short of its last events -- reads as a real difference in the
        comparison rather than as corruption.
        """
        import hashlib
        import json

        payload = json.loads(Path(manifest).read_text())
        out: Dict[str, bool] = {}
        for record in payload.get("entries", []):
            target = self.path_for(str(record["sha256"]))
            if not target.is_file():
                out[record["url"]] = False
                continue
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            out[record["url"]] = digest == str(record["sha256"])
        return out

    def offline(self) -> bool:
        """Whether every entry in the manifest is present locally."""
        manifest = self.root / MANIFEST_NAME
        if not manifest.is_file():
            return not self._entries
        return all(self.verify(manifest).values())


def _read_manifest(path: Path, root: Path) -> Dict[str, CacheEntry]:
    """Rebuild the entry table from a frozen manifest."""
    import json

    payload = json.loads(path.read_text())
    out: Dict[str, CacheEntry] = {}
    for record in payload.get("entries", []):
        digest = str(record["sha256"])
        out[str(record["url"])] = CacheEntry(
            url=str(record["url"]),
            path=root / digest[:2] / digest,
            sha256=digest,
            retrieved_utc=str(record.get("retrieved_utc", "")),
            bytes=int(record.get("bytes", 0)),
        )
    return out


def _download(url: str) -> bytes:
    """
    Fetch one URL.

    Isolated so that every network access in this module is one function, which is what
    lets a test assert an analysis touched the network exactly never.
    """
    from urllib.request import urlopen

    with urlopen(url, timeout=60) as response:
        return response.read()
