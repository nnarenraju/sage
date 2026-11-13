#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging setup and module loggers, plus run-side writers: a tensor ring
buffer, async and chunked tensor savers, and an HDF5 per-epoch loss log.

Filename        : logger.py
Description     : Short description of the file

Created on 2025-11-07 18:53:42

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

import os
import sys
import h5py
import torch
import queue
import logging
import threading

from pathlib import Path

from sage.utils.atomic_io import atomic_h5


# Attribute marking handlers Sage installed; a repeat setup_logging()
# replaces them rather than adding duplicates
_SAGE_HANDLER = "_sage_handler"

# Console format; module:lineno is shown only at DEBUG
_CONSOLE_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
_CONSOLE_FMT_DEBUG = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s"
_CONSOLE_DATEFMT = "%H:%M:%S"

# File format; always includes module:lineno
_FILE_FMT = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


def format_duration(seconds) -> str:
    """
    Duration as ``'2h 04m'``, ``'39m 12s'`` or ``'18s'``.

    Negative values become 0; fractions of a second are truncated.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
    """
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _resolve_level(level=None) -> int:
    """
    Console log level: ``level``, else ``$SAGE_LOG_LEVEL``, else INFO.

    Parameters
    ----------
    level : int or str or None
        Level number or name; a name unknown to ``logging`` is returned
        as is.

    Returns
    -------
    int
        The level number.

    Raises
    ------
    ValueError
        If ``$SAGE_LOG_LEVEL`` is not a valid level name.
    """
    if level is not None:
        return logging._nameToLevel.get(level, level) if isinstance(level, str) else level
    env = os.environ.get("SAGE_LOG_LEVEL")
    if env:
        resolved = logging._nameToLevel.get(env.strip().upper())
        if resolved is None:
            raise ValueError(
                f"SAGE_LOG_LEVEL={env!r} is not a valid level. "
                f"Use one of: {', '.join(sorted(logging._nameToLevel))}."
            )
        return resolved
    return logging.INFO


def setup_logging(export_dir=None, level=None, console: bool = True, filename: str = "run.log"):
    """
    Attach Sage's console and file log handlers, once per run.

    Handlers Sage installed earlier are removed first, so a repeat call
    does not double-log. Other libraries log WARNING and above only.

    Parameters
    ----------
    export_dir : str or Path or None
        Run directory; the log is appended to
        ``<export_dir>/logs/<filename>``. ``None`` logs to the console
        only.
    level : int or str or None
        Console level; defaults to ``$SAGE_LOG_LEVEL``, else INFO. The
        file always records Sage's DEBUG.
    console : bool
        Attach a stdout handler.
    filename : str
        Log file name within ``<export_dir>/logs/``.

    Returns
    -------
    pathlib.Path or None
        The log file path, or ``None`` if ``export_dir`` is ``None``.

    Raises
    ------
    ValueError
        If ``$SAGE_LOG_LEVEL`` is not a valid level name.
    """
    console_level = _resolve_level(level)

    root = logging.getLogger()
    # Other libraries log WARNING and above; Sage loggers pass every
    # record and the handlers filter
    root.setLevel(logging.WARNING)
    logging.getLogger("sage").setLevel(logging.DEBUG)

    # Remove handlers Sage installed earlier; others are kept
    for h in [h for h in root.handlers if getattr(h, _SAGE_HANDLER, False)]:
        root.removeHandler(h)
        h.close()

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(console_level)
        ch.setFormatter(
            logging.Formatter(
                _CONSOLE_FMT_DEBUG if console_level <= logging.DEBUG else _CONSOLE_FMT,
                datefmt=_CONSOLE_DATEFMT,
            )
        )
        setattr(ch, _SAGE_HANDLER, True)
        root.addHandler(ch)

    log_path = None
    if export_dir is not None:
        log_dir = Path(export_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / filename

        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_FILE_DATEFMT))
        setattr(fh, _SAGE_HANDLER, True)
        root.addHandler(fh)

    return log_path


def get_logger(module_name: str) -> logging.Logger:
    """
    Logger for a module; installs no handlers and opens no files.

    Until :func:`setup_logging` runs, only WARNING and above are shown,
    by Python's last-resort handler on stderr.

    Parameters
    ----------
    module_name : str
        Usually ``__name__``.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(module_name)


class TensorRingBuffer:
    """
    Fixed-capacity ring buffer of named tensor fields.

    Each field is a zero-filled ``(capacity, *shape)`` tensor on
    ``device``; once full, the oldest entries are overwritten.

    Parameters
    ----------
    capacity : int
        Number of entries held before wrapping.
    schema : dict of str to tuple
        Per-entry shape of each field, e.g. ``{"loss": (1,)}``.
    device : str
        Torch device of the buffers.
    """
    def __init__(self, capacity, schema, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.full = False

        self.buffers = {
            k: torch.zeros((capacity, *shape), device=device)
            for k, shape in schema.items()
        }

    def push(self, **kwargs):
        """
        Write one entry and advance the write pointer.

        Parameters
        ----------
        **kwargs : torch.Tensor
            Value of each field for this entry; detached before copying.

        Raises
        ------
        KeyError
            If a keyword is not a field of ``schema``.
        """
        for k, v in kwargs.items():
            self.buffers[k][self.ptr].copy_(v.detach())

        self.ptr += 1
        if self.ptr >= self.capacity:
            self.ptr = 0
            self.full = True

    def get(self):
        """
        All written entries.

        Returns
        -------
        dict of str to torch.Tensor
            The filled prefix ``[0 : ptr]`` before the buffer wraps;
            afterwards all ``capacity`` entries in storage order.
        """
        if not self.full:
            return {k: v[: self.ptr] for k, v in self.buffers.items()}
        return self.buffers


class AsyncLogger:
    """
    Logger that saves queued entries from a background thread.

    The thread saves each batch of 100 entries, and the last batch when
    closed, to ``filepath`` with ``torch.save``, overwriting the
    previous batch. Entries logged while the queue is full are dropped.

    Parameters
    ----------
    maxsize : int
        Queue capacity.
    filepath : str
        Output file, overwritten on each save.
    """

    def __init__(self, maxsize=1000, filepath="log.pt"):
        self.q = queue.Queue(maxsize=maxsize)
        self.filepath = filepath
        self.running = True

        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def log(self, data):
        """
        Queue ``data`` without blocking; drop it if the queue is full.

        Parameters
        ----------
        data : dict
            Tensors or scalars to log.
        """
        try:
            self.q.put_nowait(data)
        except queue.Full:
            pass  # drop if overloaded

    def _worker(self):
        """Drain the queue, saving each batch of 100 and the last."""
        buffer = []

        while self.running or not self.q.empty():
            try:
                item = self.q.get(timeout=0.1)
                buffer.append(item)

                if len(buffer) >= 100:
                    torch.save(buffer, self.filepath)
                    buffer.clear()

            except queue.Empty:
                continue

        if buffer:
            torch.save(buffer, self.filepath)

    def close(self):
        """Stop the thread once the queue is drained, and join it."""
        self.running = False
        self.thread.join()


class ChunkedTensorLogger:
    """
    Accumulate objects in memory and save them in fixed-size chunks.

    Each flush saves the list of objects to ``{path}_{idx}.pt`` with
    :func:`torch.save` and increments ``idx``.

    Parameters
    ----------
    chunk_size : int
        Number of items per chunk.
    path : str
        Output path prefix.
    """

    def __init__(self, chunk_size, path):
        self.chunk_size = chunk_size
        self.path = path
        self.buffer = []
        self.idx = 0

    def log(self, data):
        """
        Append ``data`` and flush once the chunk is full.

        Parameters
        ----------
        data : object
            Tensor or other picklable object.
        """
        self.buffer.append(data)

        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self):
        """Save the buffer to ``{path}_{idx}.pt``; start a new chunk."""
        torch.save(self.buffer, f"{self.path}_{self.idx}.pt")
        self.buffer = []
        self.idx += 1


class HDF5LossLogger:
    """
    Per-epoch loss log in an HDF5 file.

    Datasets ``training/loss`` and ``validation/loss`` have shape
    ``(num_epochs, len(component_names))``; :meth:`log` writes one row.

    Parameters
    ----------
    path : str or Path
        HDF5 file. An existing file with the same shape and components
        is kept, so a resumed run keeps earlier rows; otherwise it is
        created anew.
    num_epochs : int
        Total number of training epochs.
    component_names : list of str
        Loss component names, stored as each dataset's ``components``
        attribute.
    dtype : str
        NumPy dtype of the stored values (default ``"float32"``).
    """

    def __init__(self, path, num_epochs, component_names, dtype="float32"):
        self.path = path
        self.num_epochs = num_epochs
        self.component_names = list(component_names)
        self.num_components = len(self.component_names)
        self.dtype = dtype

        # Keep a compatible existing file so a resume keeps earlier rows
        if self._existing_is_compatible():
            return

        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.path, "w") as f:
            for split in ("training", "validation"):
                dataset = f.create_group(split).create_dataset(
                    "loss", shape=(num_epochs, self.num_components), dtype=dtype,
                )
                dataset.attrs["components"] = self.component_names

    def _existing_is_compatible(self):
        """
        Whether ``path`` already holds matching loss datasets.

        Both splits must have shape ``(num_epochs, num_components)`` and
        the same ``components`` attribute.

        Returns
        -------
        bool
            False also when the file is missing or unreadable.
        """
        try:
            with h5py.File(self.path, "r") as f:
                for split in ("training", "validation"):
                    if split not in f or "loss" not in f[split]:
                        return False
                    loss = f[split]["loss"]
                    if tuple(loss.shape) != (self.num_epochs, self.num_components):
                        return False
                    if [str(c) for c in loss.attrs.get("components", [])] != self.component_names:
                        return False
            return True
        except Exception:
            return False

    def log(self, loss_tensor, epoch, split):
        """
        Write one epoch's loss vector to the HDF5 file.

        Parameters
        ----------
        loss_tensor : torch.Tensor
            Shape ``(num_epochs, num_components)``; only row
            ``epoch`` is written.
        epoch : int
            Zero-based epoch index selecting the row to write.
        split : str
            Either ``"training"`` or ``"validation"``.
        """
        loss = loss_tensor[epoch].detach().cpu().numpy()

        # Atomic write: a kill mid-write leaves the file intact
        with atomic_h5(self.path) as f:
            f[split]["loss"][epoch] = loss
