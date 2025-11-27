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

# LOCAL
import logging

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
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for url in urls:
                key = (func.__module__, func.__name__, url)
                if key not in _logged_references:
                    msg = f"Reference for {func.__name__}: {url}"
                    if category:
                        msg = f"[{category}] {msg}"
                    # stacklevel=2 so that logging points to the caller
                    logger.info(msg, stacklevel=2)
                    _logged_references.add(key)
            return func(*args, **kwargs)

        return wrapper

    return decorator
