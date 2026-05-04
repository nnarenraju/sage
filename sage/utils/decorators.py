# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename         =  decorators.py
Description      =  Sage decorators

Created on 19/07/2024 at 10:20:58

__author__       =  Narenraju Nagarajan
__copyright__    =  Copyright 2024, Sage
__credits__      =  nnarenraju
__license__      =  MIT Licence
__version__      =  0.0.1
__maintainer__   =  nnarenraju
__affiliation__  =  University of Glasgow
__email__        =  nnarenraju@gmail.com
__status__       =  inUsage


Github Repository: NULL

Documentation: NULL

"""

def unreviewed_model(cls):
    """
    Class decorator that prints a warning when an unreviewed model is instantiated.

    Wraps the class constructor so that a prominent notice is printed any
    time the decorated model is created.  Used to flag experimental
    architectures that have not yet been validated on real data.

    Parameters
    ----------
    cls : type
        The model class to wrap.

    Returns
    -------
    callable
        A replacement constructor that prints the warning before delegating
        to the original class.
    """
    def mark_as_unreviewed(*args, **kwargs):
        """Print an unreviewed-model warning and construct the wrapped class."""
        print('UNREVIEWED: {} model has not be examined for issues'.format(cls.__name__))
        print('Use at your own discretion!')
        return cls(*args, **kwargs)
    return mark_as_unreviewed