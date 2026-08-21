#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : fingerprint.py
Description   : Content digests of stage products, for cascade invalidation.

Created on 2026-08-20

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

A stage's fingerprint decides whether everything downstream of it is rebuilt. It is
therefore a claim of the strongest kind a build system makes: *this stage's product is
what it was before, so nothing that consumed it needs to run again.*

A fingerprint assembled from a few hand-picked summary scalars cannot make that claim.
Summaries collide. A window lattice shifted by one sample has the same window count,
livetime and block count as the lattice it replaced; a lag ladder read out in reverse has
the same total background livetime; a FAR curve rebuilt against a different foreground
livetime has the same fitted tail shape. In every one of those cases the stage's product
changed, the fingerprint did not, and the cascade let stale downstream products stand --
the exact failure the cascade exists to prevent, arriving silently.

So a fingerprint here digests the **product**, not a description of it. The rule is that
every number a downstream stage can read must reach the digest, which makes the argument
about correctness the simple one: a consumer cannot see anything the digest did not.

Two shapes of product, two functions:

- Stages that persist a file use :func:`digest_h5`, which digests the datasets and
  attributes rather than the file's bytes. A byte digest would answer the wrong question:
  measured on h5py 3.13 / HDF5 1.14.3, writing identical data under a different chunk
  shape or with compression enabled changes the file's bytes, and a dataset created with
  ``track_times=True`` embeds a modification time that changes them on every write.
  Walking the contents is indifferent to all of it.
- Stages that rebuild their product in memory rather than persisting it -- ``segments``
  and ``grid`` -- use :func:`digest_values` over the arrays that define it. What a rebuild
  would have to reproduce is exactly what the digest covers.

Digests are truncated to 16 hex characters: 64 bits, against a campaign holding tens of
stages, is a collision probability far below the probability that the filesystem lies.

The digest deliberately does **not** cover the code that produced the product. Changing a
stage's implementation leaves its fingerprint alone if the product is unchanged, which is
the wanted behaviour for a refactor and the wrong one for a bug fix; ``--force`` is how a
bug fix invalidates, and it is a deliberate act rather than an inference.
"""

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "digest_h5",
    "digest_values",
    "combine",
    "DIGEST_CHARS",
    "VOLATILE_ATTRS",
]

#: Provenance keys excluded from :func:`digest_h5` by default.
#:
#: Every product carries a provenance block, and four of its keys move without the
#: product moving: the wall-clock time of the write, and the three that describe the code
#: that did it. Digesting them makes a stage cascade on every re-run of unchanged work,
#: which is the exact failure this module exists to avoid, arriving from the other
#: direction -- and it is worse than a missed cascade, because it is *loud*: a campaign
#: that rebuilds everything every time teaches its operator to reach for --no-cascade.
#:
#: The code is deliberately not tracked. A fingerprint answers "did the product change",
#: and a refactor that leaves a product identical has not changed it. Invalidating on a
#: bug fix is what ``--force`` is for, and is a deliberate act rather than an inference.
#:
#: Everything else in the block stays: ``spec_hash`` and ``checkpoint_sha256`` in
#: particular, since a network retrained in place changes the second and nothing else.
VOLATILE_ATTRS: Tuple[str, ...] = (
    "created_utc",
    "git_hash",
    "git_dirty",
    "sage_version",
)

# 64 bits of digest. Long enough that a collision within a campaign is not a thing that
# happens, short enough to read in a log line beside the summary scalars it follows.
DIGEST_CHARS = 16


def _feed(digest: "hashlib._Hash", value: Any, label: str = "") -> None:
    """
    Fold one value into ``digest``, recursing through containers.

    Every value is preceded by its label and its type, so ``{"a": 1}`` and ``{"a": "1"}``
    digest differently and a renamed key moves the digest. Floats go in as their IEEE-754
    bytes rather than as ``repr``, which makes ``-0.0`` and ``0.0`` distinct and does not
    depend on the platform's shortest-repr algorithm.
    """
    digest.update(label.encode("utf-8"))
    digest.update(b"\x00")
    if isinstance(value, np.ndarray):
        digest.update(f"ndarray:{value.dtype.str}:{value.shape}".encode("utf-8"))
        # ascontiguousarray, because a transposed view has the same bytes in a different
        # order and tobytes() on it would silently reorder them.
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (bytes, bytearray)):
        digest.update(b"bytes")
        digest.update(bytes(value))
    elif isinstance(value, str):
        digest.update(b"str")
        digest.update(value.encode("utf-8"))
    elif isinstance(value, bool):
        # Before the int branch: bool is a subclass of int, and True would otherwise
        # digest identically to 1.
        digest.update(b"bool")
        digest.update(b"\x01" if value else b"\x00")
    elif isinstance(value, (int, np.integer)):
        digest.update(b"int")
        digest.update(str(int(value)).encode("utf-8"))
    elif isinstance(value, (float, np.floating)):
        digest.update(b"float")
        digest.update(np.float64(value).tobytes())
    elif value is None:
        digest.update(b"none")
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=str):
            _feed(digest, value[key], f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        digest.update(f"sequence:{len(value)}".encode("utf-8"))
        for index, item in enumerate(value):
            _feed(digest, item, f"{label}[{index}]")
    else:
        raise TypeError(
            f"{label or 'value'} is a {type(value).__name__}, which has no defined "
            "digest. A fingerprint must be reproducible across processes, and the "
            "fallback for an unknown type -- repr() -- is not: it embeds object "
            "addresses. Convert it to a number, a string, or an array first"
        )


def digest_values(mapping: Mapping[str, Any]) -> str:
    """
    Digest an in-memory product.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        Named arrays and scalars that together define the product. Nested mappings and
        sequences are walked; anything else must be converted by the caller.

    Returns
    -------
    str
        Hex digest, ``DIGEST_CHARS`` characters.

    Notes
    -----
    Keys are visited in sorted order, so the digest does not depend on insertion order.
    Values inside a *list* are visited in the order given, because there the order is part
    of the product -- a lag ladder read out in reverse is a different plan.
    """
    digest = hashlib.sha256()
    _feed(digest, dict(mapping), "")
    return digest.hexdigest()[:DIGEST_CHARS]


def digest_h5(
    paths: Union[str, Path, Iterable[Union[str, Path]]],
    exclude_attrs: Sequence[str] = VOLATILE_ATTRS,
) -> str:
    """
    Digest one or more HDF5 products by their contents.

    Parameters
    ----------
    paths : path or iterable of paths
        Files to digest, in the order given. A missing file digests as its name and a
        ``missing`` marker rather than raising, so a stage that legitimately writes a
        subset of its products still produces a fingerprint -- and one that writes a
        different subset produces a different fingerprint.
    exclude_attrs : sequence of str
        Attribute names skipped at every level, defaulting to :data:`VOLATILE_ATTRS`.
        Pass an empty sequence to digest the provenance block whole.

    Returns
    -------
    str
        Hex digest, ``DIGEST_CHARS`` characters.

    Notes
    -----
    Datasets and attributes are walked in sorted name order and each contributes its full
    path, dtype, shape and raw bytes, preceded by the file's name -- so a product written
    under a different name is a different product, which is what makes a per-mode set of
    curves distinguishable. File *bytes* are deliberately not hashed: chunk shape,
    compression and ``track_times`` all move them without moving the data, and a
    fingerprint built on them would cascade the campaign on a re-write.

    h5py is imported here rather than at module scope, so that importing ``sage.search``
    stays free of it.
    """
    import h5py

    if isinstance(paths, (str, Path)):
        paths = [paths]
    digest = hashlib.sha256()
    for path in paths:
        path = Path(path)
        digest.update(f"file:{path.name}".encode("utf-8"))
        if not path.is_file():
            digest.update(b"missing")
            continue
        with h5py.File(path, "r") as handle:
            _feed_h5_group(digest, handle, frozenset(exclude_attrs))
    return digest.hexdigest()[:DIGEST_CHARS]


def _feed_h5_group(
    digest: "hashlib._Hash", group: Any, excluded: frozenset = frozenset()
) -> None:
    """
    Fold an open HDF5 group into ``digest``, depth first in sorted name order.

    Attributes are folded at every level, not only at the root: the keep threshold, the
    livetimes and the removal-mode labels all live in attrs, and a product whose arrays
    are unchanged but whose attrs are not is a different product. ``excluded`` names the
    ones that move on their own -- see :data:`VOLATILE_ATTRS`.
    """
    import h5py

    for key in sorted(group.attrs):
        if key in excluded:
            continue
        _feed(digest, _as_digestible(group.attrs[key]), f"{group.name}@{key}")
    for key in sorted(group):
        item = group[key]
        if isinstance(item, h5py.Group):
            _feed_h5_group(digest, item, excluded)
        else:
            _feed(digest, np.asarray(item[()]), item.name)


def _as_digestible(value: Any) -> Any:
    """Render an HDF5 attribute as something :func:`_feed` accepts."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return value if value.dtype.kind not in "SOU" else [str(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def combine(*parts: Any) -> str:
    """
    Render a fingerprint as readable summary fields followed by a content digest.

    Parameters
    ----------
    *parts : Any
        Fields, joined with ``:``. The last is conventionally the digest.

    Returns
    -------
    str
        The fingerprint string recorded by the stage.

    Notes
    -----
    The summary fields carry no correctness weight -- the digest already decides
    equality -- but they make a journal line legible, and a fingerprint that reads
    ``1916:191.787109:1:205:3f0c...`` says what changed in a way a bare hash does not.
    """
    return ":".join(str(part) for part in parts)
