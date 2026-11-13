#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Crash-safe HDF5 writes: atomic copy-and-replace updates, an in-place
append that moves an unreadable file aside, and an atomic rewrite.

``atomic_h5`` copies the file on every write, so it suits small files;
``append_h5`` suits large files whose contents can be regenerated.
"""

import os
import shutil
from contextlib import contextmanager

import h5py


@contextmanager
def atomic_h5(path, mode="a"):
    """
    Open a temp copy of ``path``; replace ``path`` with it on success.

    With ``mode="a"`` an existing file is copied first, so the write
    appends to its contents. On an error the temp copy is removed.

    Parameters
    ----------
    path : str or Path
        HDF5 file.
    mode : str
        ``h5py.File`` mode of the temp copy.

    Yields
    ------
    h5py.File
        The open temp copy.
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
    """
    Open ``path`` itself in append mode, without a copy.

    An unreadable file, e.g. from a kill mid-write, is moved to
    ``<path>.corrupt`` (or overwritten if it cannot be moved) and a new
    file is started.

    Parameters
    ----------
    path : str or Path
        HDF5 file.

    Yields
    ------
    h5py.File
        The open file, closed on exit.

    Raises
    ------
    BlockingIOError
        If another process holds the file open.
    """
    path = str(path)
    try:
        f = h5py.File(path, "a")
    except BlockingIOError as error:
        raise BlockingIOError(error.errno, f"{path} is open in another process; it was left as it is") from None
    except OSError:
        # Unreadable: truncated by an earlier kill, or not HDF5
        # Local import: sage.core.logger imports this module
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
    """
    Atomically write datasets and file attributes into an HDF5 file.

    A dataset of the same name is replaced, so its shape may change;
    everything else in the file is kept.

    Parameters
    ----------
    path : str or Path
        HDF5 file.
    datasets : dict
        Dataset name to array.
    attrs : dict, optional
        File attribute name to value.
    compression : str
        ``h5py`` compression filter of every dataset.
    """
    with atomic_h5(path) as f:
        for k, v in datasets.items():
            if k in f:
                del f[k]
            f.create_dataset(k, data=v, compression=compression)
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
