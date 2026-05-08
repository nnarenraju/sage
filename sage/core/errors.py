#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : errors.py
Description     : Short description of the file

Created on 2025-11-23 01:42:15

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


# LOCAL
from sage.core.logger import get_logger

logger = get_logger(__name__)


def safe_call(func, *args, fallback_return=None, **kwargs):
    """
    Call ``func`` and suppress any exception, returning a fallback value instead.

    Logs the exception at ERROR level (with function name, args, kwargs, and
    traceback message) then returns ``fallback_return`` so the caller can
    continue gracefully.

    Parameters
    ----------
    func : callable
        Function to call.
    *args
        Positional arguments forwarded to ``func``.
    fallback_return : any
        Value returned if ``func`` raises (default ``None``).
    **kwargs
        Keyword arguments forwarded to ``func``.

    Returns
    -------
    any
        The return value of ``func(*args, **kwargs)``, or ``fallback_return``
        if an exception was raised.
    """

    try:
        return func(*args, **kwargs)

    except Exception as e:
        logger.error(
            f"{func.__name__} failed.\n"
            f"  args:   {args}\n"
            f"  kwargs: {kwargs}\n"
            f"  error:  {e}"
        )
        return fallback_return
