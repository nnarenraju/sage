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

import fcntl
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    "train_runs",
    "detectors",
    "sample_rate",
    "window_samples",
    "stride_samples",
    "seed",
    "created_utc",
)

# Layout of the stamped block. Bumped whenever a key changes meaning, so a reader can
# tell an old product from a mis-stamped one.
SCHEMA_VERSION: int = 2

# How an unavailable field is recorded. A blank reads as missing wherever it surfaces --
# in an attr listing, a table, a paper -- whereas a plausible default is believed, and a
# fabricated sample rate or weight digest is exactly the failure this module exists to
# prevent.
UNRECORDED: str = ""

# Group names inside the run manifest.
_STAGE_GROUP = "stages"
_LIVETIME_GROUP = "livetime"


def _utc_now() -> str:
    """UTC stamp at second resolution, which is as fine as a campaign timeline needs."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@lru_cache(maxsize=1)
def _sage_version() -> str:
    """Installed distribution version, falling back to the subpackage's own."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("sage-gw")
    except PackageNotFoundError:
        from sage.search import __version__

        return str(__version__)


@lru_cache(maxsize=1)
def _git_state() -> Tuple[str, Any]:
    """
    Commit and working-tree cleanliness of the checkout this module was imported from.

    Cached for the life of the process: a job writes many products and the checkout it is
    running from cannot change underneath it in any way worth recording, so this costs two
    subprocesses per job rather than two per file.

    Returns
    -------
    tuple
        ``(commit, dirty)``. Both are :data:`UNRECORDED` where the code is running without
        its repository, since a copied or wheel-installed tree has no commit to name and
        guessing one would misattribute every product written from it.
    """
    root = Path(__file__).resolve().parents[2]

    def run(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return UNRECORDED, UNRECORDED
    status = run("status", "--porcelain")
    if status is None:
        return commit.strip(), UNRECORDED
    return commit.strip(), bool(status.strip())


def _path_text(value) -> str:
    """
    Render a path for provenance, mapping an unset one to :data:`UNRECORDED`.

    ``Path()`` renders as ``"."``, which would be read as the current directory rather
    than as the absent setting it is.
    """
    if value is None:
        return UNRECORDED
    text = str(value)
    return UNRECORDED if text in ("", ".") else text


def _window_geometry(spec, ckpt) -> Tuple[Any, Any]:
    """
    Sample rate and padded window length for the block, in that order.

    Taken from the geometry the spec materialises, and from the training configuration
    stored in the checkpoint where that geometry cannot yet be built. Both are
    :data:`UNRECORDED` when neither is available; the production values are not
    substituted, since the window length is what makes a recorded time interpretable.

The window conventions themselves -- 2048 Hz, 12 s plus 2x2 s of padding -- are
    :mod:`sage.search.geometry`'s and are refused if contradicted, so recording them is
    reporting a convention rather than substituting a guess. What is not recorded is a
    geometry that could not be built at all.
    """
    try:
        geometry = spec.geometry_object()
    except (NotImplementedError, ValueError, OSError):
        # A spec that defers its coalescence-time bounds to a training prior it cannot
        # reach has not resolved its geometry. OSError covers the missing prior file;
        # ValueError, a prior that does not state a usable band.
        geometry = None
    if geometry is not None:
        return float(geometry.sample_rate), int(geometry.window_samples)

    data_cfg = getattr(ckpt, "data_cfg", None) or {}
    try:
        sample_rate = float(data_cfg["sample_rate"])
        window_s = float(data_cfg["sample_length_in_s"]) + 2.0 * float(
            data_cfg["padding_length_in_s"]
        )
    except (KeyError, TypeError, ValueError):
        return UNRECORDED, UNRECORDED
    return sample_rate, int(round(sample_rate * window_s))


def _encode_attr(value):
    """
    Render one provenance value in a form an HDF5 attr can hold.

    An allowlist rather than a passthrough with one exclusion. h5py accepts a value or
    raises from inside its own write, and by then part of the block is already on the
    handle -- so anything it might reject has to be refused here, before the first write,
    for :func:`stamp` to be all-or-nothing at all. Rejecting only ``dict`` left every
    other unconvertible shape (a ragged list, a set, an arbitrary object) to fail
    mid-loop.
    """
    if value is None:
        return UNRECORDED
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        raise TypeError(
            "a provenance block is flat; a nested value cannot be stamped as an HDF5 "
            "attr and would have to be encoded, which hides it from every reader"
        )
    if isinstance(value, (str, bytes, bool, int, float, np.generic)):
        return value
    if isinstance(value, (tuple, list, np.ndarray)):
        items = [_encode_attr(item) for item in value]
        if any(isinstance(item, (list, tuple)) for item in items):
            raise TypeError(
                "a provenance value may be a flat sequence, not a nested one; an HDF5 "
                f"attr cannot hold {value!r}, and a ragged one cannot be an array at all"
            )
        kinds = {type(item) for item in items}
        if len(kinds) > 1 and not kinds <= {int, float, bool}:
            raise TypeError(
                f"a provenance sequence must be of one type; {value!r} mixes "
                f"{sorted(kind.__name__ for kind in kinds)}, which an HDF5 attr cannot "
                "hold"
            )
        return items
    raise TypeError(
        f"a provenance value must be a scalar, a string or a flat sequence of them; "
        f"cannot stamp a {type(value).__name__}"
    )


def _decode_attr(value):
    """
    Invert :func:`_encode_attr`, so a block reads back as the one that was written.

    h5py returns numpy scalars and object arrays; left as they are, a block read from a
    product would compare unequal to the block that produced it purely on type.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "ndim") and value.ndim > 0:
        return tuple(_decode_attr(item) for item in value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def _jsonable(value):
    """``json.dumps`` fallback for the types a stage report carries."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"cannot record a {type(value).__name__} in a manifest entry")


def _train_runs(ckpt) -> tuple:
    """
    Observing runs the loaded network was trained on, from its own config.

    ``UNRECORDED`` when no checkpoint was supplied or it carries no ``train_runs``. Absent
    rather than empty: an empty tuple would read as "trained on nothing", which is a
    stronger statement than "this file does not say".
    """
    if ckpt is None:
        return (UNRECORDED,)
    cfg = getattr(ckpt, "cfg", None) or {}
    runs = cfg.get("train_runs") if hasattr(cfg, "get") else None
    if not runs:
        return (UNRECORDED,)
    return tuple(str(name) for name in runs)


def searched_run_is_held_out(attrs: Dict[str, Any]) -> bool:
    """
    Whether the searched run is disjoint from the runs the network trained on.

    A search of data the network was trained on measures a background the network has
    already seen, which makes the background optimistic by an amount nothing measures.
    Read off a stamped product, so the question can be answered about a finished result
    rather than only about a running job.

    ``False`` when either side is unrecorded: an unanswerable question is not a pass.
    """
    run = str(attrs.get("observing_run", UNRECORDED))
    trained = tuple(str(name) for name in attrs.get("train_runs", ()) or ())
    if run == UNRECORDED or not trained or UNRECORDED in trained:
        return False
    return run not in trained


def provenance(spec, ckpt=None, **extra) -> Dict[str, Any]:
    """
    Build the provenance attr block written onto every output file.

    Parameters
    ----------
    spec : SearchSpec
        Configuration being run. Its :meth:`~sage.search.spec.SearchSpec.hash` is recorded
        as computed rather than re-derived downstream, because that hash is what decides
        whether a later job trusts this product or rebuilds it.
    ckpt : LoadedCheckpoint, optional
        Network actually loaded. Supplies the weight digest, the runs its noise was
        trained on, and, where the spec cannot state it alone, the window geometry the
        weights were trained under.
    **extra
        Further entries, merged last. Anything the caller knows and this cannot derive --
        a slide id, the geometry mismatches a checkpoint check reported -- belongs here,
        and an entry given twice takes the caller's value.

    Returns
    -------
    dict
        Every key in :data:`PROVENANCE_KEYS`, plus whatever ``extra`` added.
    """
    git_hash, git_dirty = _git_state()
    sample_rate, window_samples = _window_geometry(spec, ckpt)
    attrs: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sage_version": _sage_version(),
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "spec_hash": spec.hash(),
        "config_module": str(spec.config_module),
        "checkpoint_path": _path_text(
            getattr(ckpt, "path", None) or spec.engine.checkpoint
        ),
        "checkpoint_sha256": str(getattr(ckpt, "sha256", "") or UNRECORDED),
        "observing_run": str(spec.data.observing_run),
        # Recorded so the disjointness of the searched run from the trained-on runs is a
        # property of the product rather than a claim made about it. The production
        # configuration trains on O3b and searches O3a, so the background is drawn from
        # data the network never saw; that is true by construction and it should be
        # readable off any output file without going back to the checkpoint.
        "train_runs": tuple(str(name) for name in _train_runs(ckpt)),
        "detectors": tuple(str(name) for name in spec.data.detectors),
        "sample_rate": sample_rate,
        "window_samples": window_samples,
        "stride_samples": int(spec.geometry.stride_samples),
        "seed": int(spec.seed),
        "created_utc": _utc_now(),
    }
    attrs.update(extra)
    missing = [key for key in PROVENANCE_KEYS if key not in attrs]
    if missing:
        raise KeyError(f"provenance block is missing declared keys {missing}")
    return attrs


def stamp(handle, attrs: Dict[str, Any]) -> None:
    """
    Attach a provenance block to an open HDF5 handle.

    Parameters
    ----------
    handle : h5py.File or h5py.Group
        Open for writing. Passing the handle rather than a path lets the stamp be applied
        inside the same atomic write as the data, so a product can never be committed
        without it.
    attrs : dict
        A complete block, normally straight from :func:`provenance`.

    Raises
    ------
    ValueError
        A key in :data:`PROVENANCE_KEYS` is absent. A partly stamped product is worse than
        an unstamped one: it looks provenanced, so it is trusted, and the field that would
        have contradicted the reader is the one that is missing.
    """
    missing = [key for key in PROVENANCE_KEYS if key not in attrs]
    if missing:
        raise ValueError(
            f"refusing to stamp an incomplete provenance block; missing {missing}"
        )
    # Encoded in full before anything is written, so a value that cannot be held in an
    # attr leaves the handle as it was rather than half stamped.
    encoded = {key: _encode_attr(value) for key, value in attrs.items()}
    for key, value in encoded.items():
        handle.attrs[key] = value


def verify(path: str | Path, expect_spec_hash: Optional[str] = None) -> Dict[str, Any]:
    """
    Read a product's provenance and optionally assert the spec hash.

    Parameters
    ----------
    path : path
        Product to read.
    expect_spec_hash : str, optional
        Configuration the caller believes produced it, normally
        :meth:`~sage.search.spec.SearchSpec.hash` of the spec being run.

    Returns
    -------
    dict
        The stamped block, decoded to the types it was written with.

    Raises
    ------
    FileNotFoundError
        There is no product at ``path``.
    ValueError
        The product carries no complete block, or it was produced under a different
        configuration from the one asserted. Reusing a product from another configuration
        is silent and produces results that are wrong in a way nothing downstream can see,
        so the mismatch is raised rather than reported.
    """
    # Deferred: reading a product needs h5py, while the manifest and the provenance block
    # are read by tooling that should not have to load it.
    import h5py

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no product at {path}")
    with h5py.File(path, "r") as handle:
        attrs = {key: _decode_attr(value) for key, value in handle.attrs.items()}

    missing = [key for key in PROVENANCE_KEYS if key not in attrs]
    if missing:
        raise ValueError(
            f"{path} carries no usable provenance; missing {missing}. It cannot be "
            "attributed to a configuration and must be rebuilt"
        )
    if expect_spec_hash is not None and attrs["spec_hash"] != expect_spec_hash:
        raise ValueError(
            f"{path} was produced under spec {attrs['spec_hash']}, not the expected "
            f"{expect_spec_hash}; the configuration changed since it was written"
        )
    return attrs


@dataclass
class RunManifest:
    """
    Campaign-level summary: livetimes, throughput, stage completion.

    The state is the file at ``path``, not the instance. Stages run as separate jobs, so
    the process recording one stage is rarely the process that recorded the last, and a
    manifest held in memory would describe only the job that happens to be reading it.

    Each entry is stored as an HDF5 group carrying its payload as one JSON attr. Stage
    reports are heterogeneous and nested, HDF5 attrs are flat scalars and arrays, and
    coercing a report into that shape would quietly alter what a stage recorded.
    """

    path: Path

    def record_stage(self, stage: str, report: Dict[str, Any]) -> None:
        """
        Append a completed stage and its report.

        Re-running a stage replaces its entry rather than adding a second one, so the
        manifest keeps describing the products that are actually on disk.
        """
        self._record(_STAGE_GROUP, stage, report, "report")

    def drop_stage(self, stage: str) -> bool:
        """
        Remove a stage's entry, so nothing reports it as complete.

        Called before a stage re-runs. Recording only on success is not enough on its own:
        a re-run that dies part way through has already overwritten some of its products,
        but the previous entry is still on disk saying the stage completed under this
        configuration, and every downstream stage would build on a half-written product
        that looks finished. Dropping first makes the crash window read as "not run".

        Returns
        -------
        bool
            Whether an entry was there to remove.
        """
        from sage.utils.atomic_io import atomic_h5

        path = Path(self.path)
        if not path.is_file():
            return False
        lock = os.open(str(path) + ".lock", os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with atomic_h5(path) as handle:
                group = handle.get(_STAGE_GROUP)
                if group is None or stage not in group:
                    return False
                del group[stage]
        finally:
            os.close(lock)
        return True

    def record_livetime(self, run: str, coverage: Dict[str, Any]) -> None:
        """
        Store the livetime decomposition for one observing run.

        The decomposition, not a total: every rate the search quotes divides by one of
        these numbers, and which time was lost to what is the difference between a
        defensible livetime and an assertion.
        """
        self._record(_LIVETIME_GROUP, run, coverage, "coverage")

    def summary(self) -> Dict[str, Any]:
        """
        Everything needed for the methods section of the paper.

        Returns
        -------
        dict
            ``stages`` maps each recorded stage to its report and ``complete`` lists them
            in the order they were recorded; ``livetime`` maps each observing run to its
            decomposition and ``runs`` lists those.

        Notes
        -----
        Only what was actually recorded appears. A stage that never ran is absent rather
        than present with a zero, because once a zero livetime is in a table nothing
        distinguishes it from a measured one.
        """
        stages = self._entries(_STAGE_GROUP, "report")
        livetime = self._entries(_LIVETIME_GROUP, "coverage")
        return {
            "path": str(self.path),
            "stages": stages,
            "complete": tuple(stages),
            "livetime": livetime,
            "runs": tuple(livetime),
        }

    def _record(self, group_name: str, key: str, payload, attr: str) -> None:
        """
        Write one entry, under the atomic writer so a kill cannot truncate it.

        The whole read-modify-write is held under an exclusive lock on a sidecar, for the
        same reason :func:`journal` locks its append: the stages that call this run as
        separate concurrent jobs -- ``zerolag``, ``slides`` and ``injections`` all depend
        only on ``grid`` and are submitted together -- and the atomic writer replaces the
        file wholesale from a snapshot taken when it opened. Two unsynchronised writers
        therefore lose one of the two entries, or collide on the shared temporary path
        and kill the job outright.

        The lock is a sidecar rather than the manifest itself because the atomic writer
        replaces the manifest by rename, so a lock held on that inode would stop
        protecting anything the moment the first writer committed.
        """
        if "/" in key:
            raise ValueError(
                f"{key!r} cannot name a manifest entry; a slash would nest it inside "
                "another entry rather than record it"
            )
        # Deferred: h5py, via the shared atomic writer, is needed to write but not to
        # build a provenance block.
        from sage.utils.atomic_io import atomic_h5

        body = json.dumps(dict(payload), sort_keys=True, default=_jsonable)
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = os.open(str(path) + ".lock", os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with atomic_h5(path) as handle:
                group = handle.require_group(group_name)
                entry = group.require_group(key)
                if "order" not in entry.attrs:
                    # From a counter on the group, not from its current size: a writer
                    # that took its snapshot before another's entry landed would
                    # otherwise derive the same order for both, and the recorded
                    # sequence would depend on which snapshot won.
                    order = int(group.attrs.get("next_order", len(group) - 1))
                    entry.attrs["order"] = order
                    group.attrs["next_order"] = order + 1
                entry.attrs[attr] = body
                entry.attrs["recorded_utc"] = _utc_now()
        finally:
            os.close(lock)

    def _entries(self, group_name: str, attr: str) -> Dict[str, Any]:
        """Read back one group's entries, in the order they were first recorded."""
        path = Path(self.path)
        if not path.is_file():
            return {}
        import h5py

        rows: List[Tuple[int, str, Any]] = []
        with h5py.File(path, "r") as handle:
            if group_name not in handle:
                return {}
            group = handle[group_name]
            for key in group:
                entry = group[key].attrs
                rows.append((int(entry["order"]), key, json.loads(entry[attr])))
        rows.sort(key=lambda row: row[0])
        return {key: payload for _, key, payload in rows}


def journal(path: str | Path, event: Dict[str, Any]) -> None:
    """
    Append one line to the append-only stage journal.

    One JSON object per line, never rewritten. The journal is the campaign's timeline, so
    it has to survive the jobs that write it: many stages run at once, each in its own
    process, and one of them being killed must leave every line already written intact.
    Append-only line records give that, whereas a document rewritten in place loses the
    whole history to one bad write.

    Parameters
    ----------
    path : path
        Journal file, created along with its parent directory if absent.
    event : dict
        Recorded as given, with a ``utc`` stamp added when the caller has not supplied
        one: a timeline needs a time on every line.

    Notes
    -----
    The line is written with a single ``O_APPEND`` write under an exclusive lock, so
    concurrent writers interleave whole lines rather than fragments of them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(event)
    record.setdefault("utc", _utc_now())
    line = (json.dumps(record, sort_keys=True, default=_jsonable) + "\n").encode("utf-8")

    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        view = memoryview(line)
        while view:
            view = view[os.write(handle, view):]
    finally:
        os.close(handle)
