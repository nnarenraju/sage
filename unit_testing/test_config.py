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
    12. Path to random noise slice exists?
    13. Path to glitch augment should exist
    14. Path to O3b noise should exist
    15. Path to recolour PSDs should exist
    16. If network io True, model should be in permitted models
    17. Epoch testing files exist
    18. All testing files exist
    19. step_size for testing is not zero or too small
    20. trigger threshold is not too large

"""

import sys
sys.path.insert(0, ".")

import pytest
import argparse
import configs

from inspect import isclass


@pytest.fixture
def read_config():
    # Read config class from configs.py
    classes = [x for x in dir(configs) if isclass(getattr(configs, x))]
    return eval(config)





