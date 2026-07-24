#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Atomic HDF5 update for single-open writers.

Write to a private temp copy of the file, then ``os.replace`` it over the live
file in one atomic filesystem step. A crash (e.g. a SLURM TIMEOUT) mid-write
therefore leaves the live file exactly as it was -- worst case a stray ``.tmp``
that the next atomic_h5 overwrites via its snapshot copy. This mirrors
``_atomic_torch_save``/``_atomic_json_dump`` (checkpoint.py) for the ``.h5``
side products (losses.h5, validation_data.h5), so a mid-write kill can no longer
corrupt a file that is reopened in append mode every epoch and would otherwise
crash the run on its next open.

The hard-mining bank spans MANY opens per mine round, so it uses its own
multi-open transaction (``HardMiningBank.atomic_round``) rather than this
single-open helper.
"""

import os
import shutil
from contextlib import contextmanager

import h5py


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
