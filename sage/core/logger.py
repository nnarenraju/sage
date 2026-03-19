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
    module_log = Path(log_dir) / f"{module_name}.log"

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
    def __init__(self, capacity, schema, device="cpu"):
        """
        schema: dict of {name: shape}
        Example:
            {
                "loss": (1,),
                "params": (P,),
                "output": (C,),
                "target": (C,)
            }

        buffer = TensorRingBuffer(
            capacity=10000,
            schema={
                "loss": (1,),
                "params": (P,),
                "output": (C,),
                "target": (C,)
            },
            device="cpu"  # important: avoid GPU memory pressure
        )

        # inside loop
        buffer.push(
            loss=loss.detach().cpu(),
            params=signal_targets.detach().cpu(),
            output=out.detach().cpu(),
            target=targets.detach().cpu()
        )

        """
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.full = False

        self.buffers = {
            k: torch.zeros((capacity, *shape), device=device)
            for k, shape in schema.items()
        }

    def push(self, **kwargs):
        for k, v in kwargs.items():
            self.buffers[k][self.ptr].copy_(v.detach())

        self.ptr += 1
        if self.ptr >= self.capacity:
            self.ptr = 0
            self.full = True

    def get(self):
        if not self.full:
            return {k: v[: self.ptr] for k, v in self.buffers.items()}
        return self.buffers


class AsyncLogger:
    """
    Usage:
        logger = AsyncLogger()

        logger.log({
            "loss": loss.detach().cpu(),
            "params": signal_targets.detach().cpu(),
            "output": out.detach().cpu(),
            "target": targets.detach().cpu()
        })

    """

    def __init__(self, maxsize=1000, filepath="log.pt"):
        self.q = queue.Queue(maxsize=maxsize)
        self.filepath = filepath
        self.running = True

        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def log(self, data):
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
        self.running = False
        self.thread.join()


class ChunkedTensorLogger:
    def __init__(self, chunk_size, path):
        self.chunk_size = chunk_size
        self.path = path
        self.buffer = []
        self.idx = 0

    def log(self, data):
        self.buffer.append(data)

        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self):
        torch.save(self.buffer, f"{self.path}_{self.idx}.pt")
        self.buffer = []
        self.idx += 1


class HDF5LossLogger:

    def __init__(self, path, num_epochs, num_components, dtype="float32"):
        self.path = path
        self.num_epochs = num_epochs
        self.num_components = num_components
        self.dtype = dtype

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

    def log(self, loss_tensor, epoch, split):
        """
        split: "training" or "validation"
        """
        loss = loss_tensor[epoch].detach().cpu().numpy()

        with h5py.File(self.path, "a") as f:
            f[split]["loss"][epoch] = loss
