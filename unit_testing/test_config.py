# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename         =  foobar.py
Description      =  Lorem ipsum dolor sit amet

Created on 13/07/2024 at 22:06:14

__author__       =  Narenraju Nagarajan
__copyright__    =  Copyright 2024, ProjectName
__credits__      =  nnarenraju
__license__      =  MIT Licence
__version__      =  0.0.1
__maintainer__   =  nnarenraju
__affiliation__  =  University of Glasgow
__email__        =  nnarenraju@gmail.com
__status__       =  ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: https://github.com/nnarenraju/sage

Tests included:

    1. Make sure config class has all required variables
    2. Export dir path should exist
    3. Specified device must exist and have enough free space
    4. save_epoch_weight is consistant with num_epochs
    5. If pretrained, weights exist
    6. If parameter_estimation specified, do they exist in chosen model
    7. Predict learning rates are all specified epochs, are they too small or too big?
    8. Is gradient clipping too small or too big?
    9. Is num_workers available?
    10. If rescale_snr, is rescaled_snr_lower below the SNR wall?
    11. If rescale_snr, is rescaled_snr_upper too large?
    12. 



"""

import sys
sys.path.insert(0, ".")
import argparse
from configs import *

class TestORChiDConfigs:

    def __init__(self, config):
        self.test_cfg = eval(opts.config)



