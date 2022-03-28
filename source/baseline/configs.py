# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Sat Nov 27 17:09:58 2021

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

# IN-BUILT
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold

# LOCAL
from data.datasets import MLMDC1
from metrics.custom_metrics import AUC
from architectures.backend import CNN_1D
from architectures.frontend import AlphaModel, BetaModel, GammaModel
from data.transforms import Unify, Normalise, BandPass, Whiten
from losses.custom_loss_functions import BCEgw_MSEtc, regularised_BCELoss


""" DEFAULT (Lightning Baseline)"""

class Baseline:
    
    """ Data storage """
    name = "Baseline"
    # Directory to store output from pipeline/lightning
    export_dir = Path("/home/nnarenraju") / name
    
    """ Dataset Splitting """
    # Number of folds (must be at least 2, default = 5)
    n_splits = 2
    # Seed for K-Fold shuffling
    seed = 42
    # Folds are made by preserving the percentage of samples for each class
    # If set to None, dataset is split into 80-20 ratio for training and validation by default
    splitter = None
    
    """ Dataset """
    # Dataset object (opts, quick access, read only)
    dataset = MLMDC1
    dataset_params = dict()
    
    """ Architecture """
    model = GammaModel
    
    model_params = dict(
        # Simple NN Model
        model_name='mlmdc_example',
        pretrained=False,
        in_channels = 2,
        out_channels = 2,
        weights_path = 'weights.pt'
    )
    
    """ Epochs and Batches """
    num_steps = 25000
    num_epochs = 25
    # Equal batch_size for both training and validation
    batch_size = 100 
    # every 'n' epochs
    save_freq = 5
    
    """ Gradient Clipping """
    # Clip gradients to make convergence somewhat easier
    clip_norm = 100
    
    """ Optimizer """
    optimizer = optim.Adam
    optimizer_params = dict(lr=5e-5, weight_decay=1e-6)
    
    """ Scheduler """
    scheduler = None
    scheduler_params = dict()
    scheduler_target = None
    batch_scheduler = False
    
    """ Loss Function """
    # Add the () as suffix to loss function, returns object instead
    loss_function = regularised_BCELoss(dim=2)
    output_loss_file = "losses.txt"
    
    """ Evaluation Metric """
    eval_metric = None
    # Normalised threshold for accuracy
    accuracy_thresh = 0.5
    
    """ Storage Devices """
    store_device = 'cpu'
    train_device = 'cpu'
    
    """ Data Transforms """
    # Input to Unfy should always be a list
    # Input to transforms should be a dict
    transforms = dict(
        train=Unify([
            BandPass(lower=16, upper=512, fs=2048., order=6),
            Whiten(max_filter_duration=0.25, trunc_method='hann',
                   remove_corrupted=True, low_frequency_cutoff=15., sample_rate=2048.),
        ]),
        test=None,
        target=None
    )
    
    # Debugging (size: train_data = 1e4, val_data = 1e3)
    debug = False
    
    
""" MANUAL BASELINE """

class ManualBaseline:
    
    """ Data storage """
    name = "ManualBaseline"
    # Directory to store output from pipeline
    export_dir = Path("/home/nnarenraju")
    
    """ Dataset Splitting """
    # Folds are made by preserving the percentage of samples for each class
    splitter = None
    
    """ Dataset """
    # Dataset object (opts, quick access, read only)
    dataset = MLMDC1
    dataset_params = dict()
    
    """ Architecture """
    model = None
    
    model_params = dict(
        # Simple NN Model
        model_name='mlmdc_manual_example',
        pretrained=False,
        in_channels = 2,
        out_channels = 1
    )
    
    """ Epochs and Batches """
    num_steps = 25000
    num_epochs = 50
    batch_size = 4
    
    """ Loss Function """
    # Add the () as suffix to loss function, returns object instead
    loss_function = None
    
    """ Data Transforms """
    # Input to Unfy should always be a list
    # Input to transforms should be a dict
    transforms = dict(
        train=Unify([
            Normalise(factors=[1.8021542328645444e-19, 9.216145461527009e-20]),
        ]),
        test=Unify([
            Normalise(factors=[1.8021542328645444e-19, 9.216145461527009e-20]),
        ]),
        target=None
    )
    
    # Debugging (size: train_data = 1e4, val_data = 1e3)
    debug = False
    
    
    
    
""" CUSTOM MODELS FOR EXPERIMENTATION """

class LonelyTimm:
    
    """ Name and Path """
    name = "LonelyTimm"
    export_dir = ""
    
    """ Dataset Splitting """
    # Number of folds (must be at least 2, default = 5)
    n_splits = 5
    # Seed for K-Fold shuffling
    seed = 150914
    # Folds are made by preserving the percentage of samples for each class
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    
    """ Dataset """
    # Dataset object (opts, quick access, read only)
    dataset = MLMDC1
    dataset_params = dict()
    
    """ Architecture """
    model = BetaModel
    
    model_params = dict(
        # Timm Model
        model_name='resnet34',
        timm_params={'in_chans':3},
        pretrained=False,
        num_classes=1,
        # Trainable Backend
        backend=CNN_1D,
        backend_params=dict(
            in_channels = 3,
            out_channels = 3,
            force_out_size = 128,
            check_out_size = True
        )
    )
    
    """ Epochs and Batches """
    num_steps = 25000
    num_epochs = 15
    batch_size = 8
    
    """ Optimizer """
    optimizer = optim.Adam
    optimizer_params = dict(lr=1e-4, weight_decay=1e-6)
    
    """ Scheduler """
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    scheduler_target = None
    batch_scheduler = False
    
    """ Loss Function """
    loss_function = nn.BCEWithLogitsLoss()
    
    """ Evaluation Metric """
    eval_metric = AUC().torch
    
    """ Data Transforms """
    transforms = dict(
        train=Unify([
            Normalise(factors=[1.0, 1.0]),
            BandPass(lower=12, upper=512),
        ]),
        test=Unify([
            Normalise(factors=[1.0, 1.0]),
            BandPass(lower=12, upper=512),
        ]),
        target=None
    )
    
    # Debugging (size: train_data = 1e4, val_data = 1e3)
    debug = False
