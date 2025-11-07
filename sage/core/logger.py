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
import logging
import sys
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
