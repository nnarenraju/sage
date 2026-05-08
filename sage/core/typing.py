#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : base_entities.py
Description     : Short description of the file

Created on 2025-11-06 17:57:46

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
import torch
import numpy as np

from typing import Union


# Speed is key
TorchArray = Union[torch.Tensor]

# Structured dtype for segment results
SEGMENT_DTYPE = np.dtype(
    [
        ("detector", "U2"),  # Detector, e.g. 'H1'
        ("flag", "U10"),  # Data quality flag, e.g. 'H1_DATA'
        ("start_time", "f8"),  # Start time (float)
        ("end_time", "f8"),  # End time (float)
        ("observing_run", "U10"),  # Run name
        ("segments", "O"),  # Segment list or sub-array
    ]
)
