#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calls that log a failure and return a fallback value instead of
raising.

Filename        : errors.py
Description     : Short description of the file

Created on 2025-11-23 01:42:15

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

from sage.core.logger import get_logger

logger = get_logger(__name__)


def safe_call(func, *args, fallback_return=None, **kwargs):
    """
    ``func(*args, **kwargs)``, or ``fallback_return`` if it raises.

    A failure is logged at ERROR level with the function name, the
    arguments and the exception.

    Parameters
    ----------
    func : callable
        Function to call.
    *args
        Positional arguments forwarded to ``func``.
    fallback_return : any
        Value returned if ``func`` raises.
    **kwargs
        Keyword arguments forwarded to ``func``.

    Returns
    -------
    any
        The result of ``func``, or ``fallback_return`` if it raised.
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
