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
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold

# LOCAL
from metrics.custom_metrics import AUC
from data.datasets import MLMDC1
from architectures.backend import CNN_1D
from architectures.frontend import AlphaModel, BetaModel
from data.transforms import Unify, Normalise, BandPass


""" DEFAULT """
class Baseline:
    
    """ Data storage """
    name = "Baseline"
    # Directory to store output from pipeline/lightning
    export_dir = ""
    
    """ Dataset Splitting """
    # Number of folds (must be at least 2, default = 5)
    n_splits = 2
    # Seed for K-Fold shuffling
    seed = 150914
    # Folds are made by preserving the percentage of samples for each class
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    
    """ Dataset """
    # Dataset object (opts, quick access, read only)
    dataset = MLMDC1
    dataset_params = dict()
    
    """ Architecture """
    model = AlphaModel
    
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
            num_channels = (32, 64, 128),
            kernel_size = 3, 
            stride = 2,
            force_out_size = 128,
            check_out_size = True
        )
    )
    
    """ Epochs and Batches """
    num_steps = 25000
    num_epochs = 5
    batch_size = 8
    
    """ Optimizer """
    optimizer = optim.Adam
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)
    
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
    # Input to Unfy should always be a list
    # Input to transforms should be a dict
    transforms = dict(
        train=Unify([
            Normalise(factors=[1.0, 1.0]),
            BandPass(lower=12, upper=512),
        ]),
        test=Unify([
            Normalise(factors=[1.0, 1.0]),
            BandPass(lower=12, upper=512),
        ])
    )
    
    # Debugging (size: train_data = 1e4, val_data = 1e3)
    debug = False



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
        ])
    )
    
    # Debugging (size: train_data = 1e4, val_data = 1e3)
    debug = False
