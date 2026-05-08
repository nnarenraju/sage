#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : config.py
Description     : Short description of the file

Created on 2026-03-16 11:45:28

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

# Packages
import torch


class O3aCFG:

    export_dir = "./run_export"
    batch_size = 1024
    device = "cuda:1"
    dtype = torch.float32
    detectors = ["H1", "L1"]
    do_point_estimate = ["tc", "mchirp"]
    autocast = True
    class_balance = 0.5
    clip_norm = 1.0
    num_epochs = 5  # 400
    training_iterations = 200_000
    validation_iterations = 200_000


class O3aDataCFG:

    data_dir = "/local/scratch/igr/nnarenraju/o3a/data_dir"
    training_noise_files = [
        "/local/scratch/igr/nnarenraju/o3a/data_release/data_H1_O3a.bin",
        "/local/scratch/igr/nnarenraju/o3a/data_release/data_L1_O3a.bin",
    ]
    validation_noise_files = [
        "/local/scratch/igr/nnarenraju/o3a/data_release/data_H1_O3b.bin",
        "/local/scratch/igr/nnarenraju/o3a/data_release/data_L1_O3b.bin",
    ]
    sample_rate = 2048.0  # Hz
    noise_low_frequency_cutoff = 15.0  # Hz
    signal_low_frequency_cutoff = 20.0  # Hz
    sample_length_in_s = 12.0  # seconds
    padding_length_in_s = 2.0  # seconds
