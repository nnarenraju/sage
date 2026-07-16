#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : logger.py
Description     : Short description of the file

Created on 2025-11-07 18:53:42

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Logging essentials
import sys
import h5py
import torch
import queue
import logging
import threading

from pathlib import Path


def setup_logging(log_dir: str = "logs", level: int = logging.INFO):
    """
    Configure global and per-module logging.

    Args:
        log_dir (str): Directory where log files are stored.
        level (int): Minimum logging level.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Formatter for all logs
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Main log file (all logs) ---
    main_log = log_dir / "main.log"
    main_handler = logging.FileHandler(main_log, mode="a")
    main_handler.setFormatter(formatter)
    main_handler.setLevel(level)

    # --- Stream handler (console) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Configure root logger (collects everything)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers when reloading
    if not root_logger.handlers:
        root_logger.addHandler(main_handler)
        root_logger.addHandler(console_handler)


def get_logger(module_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Get a logger for a specific module.
    Each module has its own log file + logs also go to the main file.

    Args:
        module_name (str): Name of the module.
        log_dir (str): Directory where log files are stored.

    Returns:
        logging.Logger: Configured logger instance
    """

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # Per-module log file
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    module_log = log_path / f"{module_name}.log"

    if not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(module_log)
        for h in logger.handlers
    ):
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(module_log, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


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

        with h5py.File(self.path, "a") as f:
            f[split]["loss"][epoch] = loss
