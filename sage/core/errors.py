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
    """Safe call helper: call func without exit
    On exception, print the error and return fallback_return

    Args:
        func (_type_): _description_
        fallback_return (_type_): value to return if exception occurs

    Returns:
        _type_: _description_
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
