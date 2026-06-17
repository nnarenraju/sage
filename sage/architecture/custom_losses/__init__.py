# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Fri Jan 28 19:09:03 2022

__author__      = nnarenraju
__copyright__   = Copyright 2021, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

from .loss_functions import BCEWithPEregLoss, BCEWithPEsigmaLoss
from .consistency_loss import ConsistencyNLLLoss
from .gradient_balancer import GradientNormBalancer

__all__ = [
    "BCEWithPEregLoss",
    "BCEWithPEsigmaLoss",
    "ConsistencyNLLLoss",
    "GradientNormBalancer",
]
