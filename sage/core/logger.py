#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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

# Logging essentials
import os
import sys
import h5py
import torch
import queue
import logging
import threading

from pathlib import Path

from sage.utils.atomic_io import atomic_h5


# Marker attribute stamped on handlers Sage installs, so setup_logging() can
# be called again (a resumed segment, a notebook re-run) without stacking
# duplicate handlers and printing everything twice.
_SAGE_HANDLER = "_sage_handler"

# Console: readable at a glance. No module:lineno at INFO -- that detail is
# noise when you are watching a run, and it is always in the file anyway.
_CONSOLE_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
_CONSOLE_FMT_DEBUG = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s"
_CONSOLE_DATEFMT = "%H:%M:%S"

# File: full provenance, always. The file is for reading after the fact.
_FILE_FMT = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


def format_duration(seconds) -> str:
    """Render a duration as ``'2h 04m'``, ``'39m 12s'`` or ``'18s'``.

    Used for epoch timings and ETAs so long runs read naturally in the log
    rather than as a raw float.
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
    """Resolve the console level: explicit arg, then $SAGE_LOG_LEVEL, then INFO."""
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
    Attach Sage's log handlers. Call this **once**, at the start of a run.

    This is the only function that configures handlers. :func:`get_logger` is
    deliberately inert, so importing a Sage module never touches the
    filesystem and never emits anything.

    Parameters
    ----------
    export_dir : str or Path or None
        The run's export directory. Logs are written to
        ``<export_dir>/logs/<filename>``, so they live with the run they
        describe rather than in whatever directory the job happened to start
        in. When ``None``, no file is written and logging goes to the console
        only -- appropriate for notebooks and one-off scripts.
    level : int or str or None
        Console verbosity. Defaults to ``$SAGE_LOG_LEVEL`` if set, else
        ``INFO``. The *file* always records DEBUG regardless, so turning the
        console down never loses detail you might need afterwards.
    console : bool
        Attach a stdout handler. Disable for jobs where stdout is captured
        separately.
    filename : str
        Log file name within ``<export_dir>/logs/``.

    Returns
    -------
    pathlib.Path or None
        The log file path, or ``None`` if ``export_dir`` was not given.

    Notes
    -----
    Safe to call more than once: previously installed Sage handlers are
    removed first, so a resumed run does not double-log.
    """
    console_level = _resolve_level(level)

    root = logging.getLogger()
    # Root must pass DEBUG through; the handlers decide what is actually shown.
    root.setLevel(logging.DEBUG)

    # Drop handlers we installed previously (leave anyone else's alone).
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
    Return the logger for a module. Has no side effects.

    Safe to call at module scope -- it creates no directories, opens no files
    and installs no handlers. Until :func:`setup_logging` runs, records simply
    go nowhere (Python's last-resort handler still surfaces WARNING and above
    on stderr, so genuine problems are never silently swallowed).

    This used to create a ``logs/`` directory and a per-module log file at call
    time. Because it is called at module scope throughout the package, merely
    importing Sage scattered ``logs/`` directories into whatever directory the
    caller happened to be in, and would fail outright on a read-only one.

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
    Fixed-capacity ring buffer that stores named tensor fields.

    Pre-allocates a contiguous tensor for each named field and overwrites
    the oldest entries once the buffer is full.  All data is kept on
    ``device`` (use ``"cpu"`` to avoid GPU memory pressure).

    Parameters
    ----------
    capacity : int
        Maximum number of entries the buffer holds before wrapping.
    schema : dict[str, tuple]
        Mapping from field name to per-entry shape.  Example::

            {
                "loss":   (1,),
                "params": (P,),
                "output": (C,),
                "target": (C,),
            }
    device : str
        Torch device string for the pre-allocated tensors.

    Example
    -------
    .. code-block:: python

        buffer = TensorRingBuffer(
            capacity=10000,
            schema={"loss": (1,), "params": (P,), "output": (C,), "target": (C,)},
            device="cpu",
        )
        # inside training loop
        buffer.push(
            loss=loss.detach().cpu(),
            params=signal_targets.detach().cpu(),
            output=out.detach().cpu(),
            target=targets.detach().cpu(),
        )
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
        Write one entry to the buffer, advancing the write pointer.

        Parameters
        ----------
        **kwargs : torch.Tensor
            One keyword per field defined in ``schema``.  Each tensor is
            detached before copying so no gradient is accidentally stored.
        """
        for k, v in kwargs.items():
            self.buffers[k][self.ptr].copy_(v.detach())

        self.ptr += 1
        if self.ptr >= self.capacity:
            self.ptr = 0
            self.full = True

    def get(self):
        """
        Return all valid entries.

        Returns
        -------
        dict[str, torch.Tensor]
            If the buffer has not yet wrapped, returns only the filled
            prefix ``[0 : ptr]``.  Once full, returns all ``capacity``
            entries (oldest-first order is not guaranteed after wrapping).
        """
        if not self.full:
            return {k: v[: self.ptr] for k, v in self.buffers.items()}
        return self.buffers


class AsyncLogger:
    """
    Non-blocking logger that offloads disk I/O to a background thread.

    Incoming data dicts are placed on an in-memory queue; a daemon thread
    drains the queue in batches of 100 and serialises them to ``filepath``
    with ``torch.save``.  Excess entries are silently dropped when the
    queue is full, so the training loop is never blocked.

    Parameters
    ----------
    maxsize : int
        Maximum number of pending log entries before drops occur.
    filepath : str
        Path where batched entries are saved (overwritten each flush).

    Example
    -------
    .. code-block:: python

        logger = AsyncLogger()
        logger.log({
            "loss":   loss.detach().cpu(),
            "params": signal_targets.detach().cpu(),
            "output": out.detach().cpu(),
            "target": targets.detach().cpu(),
        })
        logger.close()  # flush remaining entries and join the thread
    """

    def __init__(self, maxsize=1000, filepath="log.pt"):
        self.q = queue.Queue(maxsize=maxsize)
        self.filepath = filepath
        self.running = True

        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def log(self, data):
        """
        Submit a data dict to the logging queue (non-blocking).

        If the queue is full, the entry is silently discarded rather than
        blocking the caller.

        Parameters
        ----------
        data : dict
            Arbitrary dictionary of tensors or scalars to log.
        """
        try:
            self.q.put_nowait(data)
        except queue.Full:
            pass  # drop if overloaded

    def _worker(self):
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

    def close(self):
        """Flush remaining entries and join the background thread."""
        self.running = False
        self.thread.join()


class ChunkedTensorLogger:
    """
    Accumulate tensors in memory and flush to disk in fixed-size chunks.

    Each flush writes a Python list of tensors to ``{path}_{idx}.pt``
    (via :func:`torch.save`) and increments the chunk index.

    Parameters
    ----------
    chunk_size : int
        Number of items to accumulate before automatically flushing.
    path : str
        File-path prefix for the output files (suffix ``_<idx>.pt`` is
        appended automatically).
    """

    def __init__(self, chunk_size, path):
        self.chunk_size = chunk_size
        self.path = path
        self.buffer = []
        self.idx = 0

    def log(self, data):
        """
        Append *data* to the buffer and flush if the chunk is full.

        Parameters
        ----------
        data : any
            Tensor or other pickleable object to accumulate.
        """
        self.buffer.append(data)

        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self):
        """Write the current buffer to disk and reset state for the next chunk."""
        torch.save(self.buffer, f"{self.path}_{self.idx}.pt")
        self.buffer = []
        self.idx += 1


class HDF5LossLogger:
    """
    Persistent, epoch-indexed loss logger backed by an HDF5 file.

    Pre-allocates datasets of shape ``(num_epochs, num_components)`` for
    both the ``"training"`` and ``"validation"`` splits at construction
    time, then writes one row per epoch via :meth:`log`.

    The resulting file can be read directly with ``h5py``::

        with h5py.File("losses.h5", "r") as f:
            train_loss = f["training"]["loss"][:]   # (E, C) float32
            val_loss   = f["validation"]["loss"][:]

    Parameters
    ----------
    path : str
        File path for the HDF5 output. A compatible existing file (same
        ``num_epochs``/``num_components``) is PRESERVED and appended to, so a
        chained-segment resume keeps prior epochs' loss rows; only an absent or
        shape-mismatched file is (re)created fresh.
    num_epochs : int
        Total number of training epochs (pre-allocates the dataset).
    num_components : int
        Number of scalar loss components logged per epoch (e.g. 4 for
        :class:`BCEWithPEsigmaLoss`: total, BCE, reg, coupling).
    dtype : str
        NumPy dtype string for the stored values (default ``"float32"``).
    """

    def __init__(self, path, num_epochs, num_components, dtype="float32"):
        self.path = path
        self.num_epochs = num_epochs
        self.num_components = num_components
        self.dtype = dtype

        # Resume-safe: if a compatible file already exists (same shape), KEEP it
        # so loss rows written by earlier chained 2-day segments are preserved.
        # Opening 'w' unconditionally (the old behaviour) truncated the file on
        # every resume, permanently zeroing all prior epochs' loss history.
        if self._existing_is_compatible():
            return

        with h5py.File(self.path, "w") as f:

            # Train group
            train_grp = f.create_group("training")
            train_grp.create_dataset(
                "loss",
                shape=(num_epochs, num_components),
                dtype=dtype,
            )

            # Validation group
            val_grp = f.create_group("validation")
            val_grp.create_dataset(
                "loss",
                shape=(num_epochs, num_components),
                dtype=dtype,
            )

    def _existing_is_compatible(self):
        """True iff ``self.path`` already holds training/validation loss datasets
        of the expected ``(num_epochs, num_components)`` shape (i.e. this is a
        resume of the same run and the history must be preserved)."""
        try:
            with h5py.File(self.path, "r") as f:
                for split in ("training", "validation"):
                    if split not in f or "loss" not in f[split]:
                        return False
                    if tuple(f[split]["loss"].shape) != (
                        self.num_epochs, self.num_components
                    ):
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
            Shape ``(num_epochs, num_components)``.  Only row ``epoch`` is
            written; the rest are ignored.  This matches the
            ``loss_components`` tensor stored on training/validation objects.
        epoch : int
            Zero-based epoch index selecting the row to write.
        split : str
            Either ``"training"`` or ``"validation"``.
        """
        loss = loss_tensor[epoch].detach().cpu().numpy()

        # Crash-atomic: this file is reopened in append mode every epoch, so a
        # mid-write kill that corrupted it would crash the run on its next open.
        with atomic_h5(self.path) as f:
            f[split]["loss"][epoch] = loss
