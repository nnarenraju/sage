#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Decorators: log a function's references once per session, and trim
the corrupted edges of a returned segment.

Filename        : decorators.py
Description     : Short description of the file

Created on 2025-11-27 01:41:10

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

import functools
import numpy as np

from functools import wraps
from dataclasses import dataclass

from sage.core.logger import get_logger

logger = get_logger(__name__)
# References already logged in this session
_logged_references = set()


def reference(*urls, category=None):
    """
    Decorator that logs a function's references once per session.

    Parameters
    ----------
    *urls : str
        Reference URLs or identifiers.
    category : str, optional
        Label such as "paper", "code" or "documentation".

    Returns
    -------
    callable
        The decorator.

    Examples
    --------
    >>> @reference("https://pycbc.org/", category="code")
    ... def resample(x): ...
    """

    def decorator(func):
        """Wrap ``func`` to log its references."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Log unseen references, then call ``func``."""
            for url in urls:
                key = (func.__module__, func.__name__, url)
                if key not in _logged_references:
                    msg = f"Ref. for {func.__name__} in module {func.__module__}: {url}"
                    if category:
                        msg = f"[{category}] {msg}"
                    # stacklevel=2 attributes the record to the caller
                    logger.info(msg, stacklevel=2)
                    _logged_references.add(key)
            return func(*args, **kwargs)

        return wrapper

    return decorator


@dataclass(frozen=True)
class CorruptionBudget:
    """
    Corrupted duration at each edge of a processed segment.

    Filters, resampling and Welch windowing corrupt both edges; the
    ``corruption`` decorator trims them.

    Attributes
    ----------
    left : float
        Corrupted duration at the start of the segment (seconds).
    right : float
        Corrupted duration at the end of the segment (seconds).
    total : float
        ``left + right`` (property).
    """

    left: float  # seconds
    right: float  # seconds

    @property
    def total(self):
        """
        Total corrupted duration, ``left + right``, in seconds.

        Returns
        -------
        float
        """
        return self.left + self.right


def corruption(budget: CorruptionBudget):
    """
    Decorator that trims the corrupted edges given by ``budget``.

    The decorated function returns a NumPy array and takes a
    ``data_config`` keyword argument with a ``sample_rate`` attribute.

    Parameters
    ----------
    budget : CorruptionBudget
        Corrupted seconds at each edge.

    Returns
    -------
    callable
        The decorator.

    Raises
    ------
    ValueError
        If ``data_config`` or its ``sample_rate`` is missing, or the
        budget covers the whole segment.
    TypeError
        If the function does not return a NumPy array.
    """

    def decorator(func):
        """Wrap ``func`` to trim its returned segment."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Call ``func`` and trim the corrupted edges."""
            data_cfg = kwargs.get("data_config", None)
            if data_cfg is None:
                raise ValueError("data_config must be provided to apply corruption")

            sample_rate = getattr(data_cfg, "sample_rate", None)
            if sample_rate is None:
                raise ValueError("data_config must have a sample_rate attribute")

            ts = func(*args, **kwargs)
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
