#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : decorators.py
Description     : Short description of the file

Created on 2025-11-27 01:41:10

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


# General
import functools
import numpy as np

from functools import wraps
from dataclasses import dataclass

# LOCAL
from sage.core.logger import get_logger

logger = get_logger(__name__)
# Keep track of references logged in this session
_logged_references = set()


def reference(*urls, category=None):
    """
    Decorator to log references for a function.

    Parameters
    ----------
    *urls : str
        One or more reference URLs or identifiers.
    category : str, optional
        Category label for the reference
        (e.g., "paper", "code", "documentation").

    Example usage:
        @reference(
            os.path.join(PYCBC_PARENT_URL, "pycbc/filter/resample.html"),
            os.path.join(PYCBC_PARENT_URL, "pycbc/types/array.html#Array.roll"),
            category="documentation",
        )
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for url in urls:
                key = (func.__module__, func.__name__, url)
                if key not in _logged_references:
                    msg = f"Ref. for {func.__name__} in module {func.__module__}: {url}"
                    if category:
                        msg = f"[{category}] {msg}"
                    # stacklevel=2 so that logging points to the caller
                    logger.info(msg, stacklevel=2)
                    _logged_references.add(key)
            return func(*args, **kwargs)

        return wrapper

    return decorator


@dataclass(frozen=True)
class CorruptionBudget:
    left: float  # seconds
    right: float  # seconds

    @property
    def total(self):
        return self.left + self.right


def corruption(budget: CorruptionBudget):
    """
    Decorator to trim left/right edges of a time series according to CorruptionBudget.
    Expects a `data_config` keyword argument with a `sample_rate` attribute.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Expect data_config in kwargs
            data_cfg = kwargs.get("data_config", None)
            if data_cfg is None:
                raise ValueError("data_config must be provided to apply corruption")

            sample_rate = getattr(data_cfg, "sample_rate", None)
            if sample_rate is None:
                raise ValueError("data_config must have a sample_rate attribute")

            ts = func(*args, **kwargs)  # original segment
            if not isinstance(ts, np.ndarray):
                raise TypeError(
                    "Corruption decorator expects function to return np.ndarray"
                )

            left_samples = int(budget.left * sample_rate)
            right_samples = int(budget.right * sample_rate)

            if left_samples + right_samples >= len(ts):
                raise ValueError("CorruptionBudget exceeds segment length")

            return ts[left_samples : len(ts) - right_samples]

        return wrapper

    return decorator
