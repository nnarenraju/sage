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
    # Decorator to mark unreviewed models
    def mark_as_unreviewed(*args, **kwargs):
        print('UNREVIEWED: {} model has not be examined for issues'.format(cls.__name__))
        print('Use at your own discretion!')
        return cls(*args, **kwargs)
    return mark_as_unreviewed