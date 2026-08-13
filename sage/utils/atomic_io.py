#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Atomic HDF5 update for single-open writers.

Write to a private temp copy of the file, then ``os.replace`` it over the live
file in one atomic filesystem step. A crash (e.g. a SLURM TIMEOUT) mid-write
therefore leaves the live file exactly as it was -- worst case a stray ``.tmp``
that the next atomic_h5 overwrites via its snapshot copy. This mirrors
``_atomic_torch_save``/``_atomic_json_dump`` (checkpoint.py) for the small ``.h5``
side products, so a mid-write kill can no longer corrupt a file that is reopened
in append mode every epoch and would otherwise crash the run on its next open.

Choosing between the two helpers here
-------------------------------------
``atomic_h5`` costs a full copy of the file per write, so it is only worth it
while the file stays small. ``losses.h5`` is a few KB and holds the whole loss
curve, so it is written atomically. ``validation_data.h5`` grows to ~300 MB over
a 128-epoch run, where a per-epoch snapshot would move tens of GB per run for a
regenerable diagnostic; it uses :func:`append_h5` instead, which writes in place
and quarantines an unreadable file rather than copying to avoid ever producing
one. That keeps the property that actually matters -- a killed write can never
crash the *next* segment on open -- without the I/O.

The hard-mining bank spans MANY opens per mine round, so it uses its own
multi-open transaction (``HardMiningBank.atomic_round``) rather than this
single-open helper.
"""

import os
import shutil
from contextlib import contextmanager

import h5py

# NOTE: no module-level sage.core.logger import -- logger.py imports atomic_h5
# from here, so that would be a cycle. append_h5 imports it lazily instead.


@contextmanager
def atomic_h5(path, mode="a"):
    """Yield an ``h5py.File`` on a temp copy of ``path``; atomically commit on
    clean exit, discard on any error/kill.

    With ``mode='a'`` (the default) the existing file is snapshotted first, so
    prior groups/datasets are preserved and only the new write is added -- the
    same append semantics as ``h5py.File(path, 'a')``, but crash-atomic.
    """
    path = str(path)
    tmp = path + ".tmp"
    if mode == "a" and os.path.exists(path):
        shutil.copy2(path, tmp)          # snapshot committed state to append onto
    committed = False
    try:
        with h5py.File(tmp, mode) as f:
            yield f
        os.replace(tmp, path)            # atomic commit
        committed = True
    finally:
        if not committed and os.path.exists(tmp):
            try:
                os.remove(tmp)           # discard the partial write
            except OSError:
                pass


@contextmanager
def append_h5(path):
    """Yield an ``h5py.File`` opened in append mode on ``path`` itself -- no
    snapshot copy, so the cost is the new data only.

    For files large enough that a per-write copy is the dominant I/O and whose
    contents can be regenerated (``validation_data.h5``). A kill during the write
    can leave the file unreadable, which ``atomic_h5`` prevents outright; here it
    is instead handled on the next open, by moving the damaged file aside to
    ``<path>.corrupt`` and starting a fresh one. The run therefore survives a
    mid-write kill either way -- it loses the earlier epochs' diagnostics rather
    than paying to copy them every epoch.
    """
    path = str(path)
    try:
        f = h5py.File(path, "a")
    except OSError:
        # Unreadable: truncated by an earlier kill, or not HDF5 at all. Keep the
        # evidence, do not let it take the run down on open.
        from sage.core.logger import get_logger

        quarantine = path + ".corrupt"
        try:
            os.replace(path, quarantine)
            get_logger(__name__).warning(
                "%s was unreadable (likely killed mid-write); moved to %s and "
                "started a new file.", path, quarantine,
            )
        except OSError:
            get_logger(__name__).warning(
                "%s was unreadable and could not be moved aside; overwriting.", path
            )
        f = h5py.File(path, "w")
    try:
        yield f
    finally:
        f.close()


def write_h5(path, datasets, attrs=None, compression="gzip"):
    """Crash-atomically (re)write an HDF5 file with the given datasets/attrs.

    A dataset of the same name is replaced rather than merged, so a rewrite with a
    different shape succeeds instead of raising. Everything else already in the file is
    preserved, since ``atomic_h5`` snapshots before writing.

    Shared by the validation/testing side products and by the search's trigger shards and
    manifests, so one definition of "written safely" covers both.
    """
    with atomic_h5(path) as f:
        for k, v in datasets.items():
            if k in f:
                del f[k]
            f.create_dataset(k, data=v, compression=compression)
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
