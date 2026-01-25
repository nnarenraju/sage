#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : helper.py
Description     : Short description of the file

Created on 2026-01-23 03:24:33

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# There are too many constants in the Phenom files which need to be tensors
# Creating tensors during a hot-path iteration kills the torch graph
# Here, we store lots of these constants and allow for device setting
# Putting Phenom into a massive class is not torch.compile friendly (can't use self)
# Dataclass with frozen=True is fine too

class PhenomConstants:


