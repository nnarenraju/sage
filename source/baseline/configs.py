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
from data.datasets import MLMDC1, BatchLoader
from metrics.custom_metrics import AUC
from architectures.backend import CNN_1D
from architectures.frontend import AlphaModel, BetaModel, GammaModel, KappaModel
from data.transforms import Unify, Normalise, BandPass, Whiten, MultirateSampling
from losses.custom_loss_functions import BCEgw_MSEtc, regularised_BCELoss


""" DEFAULT (Lightning Baseline)"""

class Baseline:
    
    """ Data storage """
    name = "Baseline_longer"
    # Directory to store output from pipeline/lightning
    export_dir = Path("/Users/nnarenraju/Desktop") / name
    
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
    # if dataset object is set to 'simple', trainable.hdf should exist
    dataset = MLMDC1
    dataset_params = dict()
    
    """ Architecture """
    model = GammaModel
    
    model_params = dict(
        # Simple NN Model
        model_name = 'mlmdc_example',
        in_channels = 2,
        out_channels = 2,
        store_device = 'cpu',
    )
    
    # Pre-trained weights (does not refer to pretrained timm models)
    # This refers to a pretrained frontend+backend model on GW data
    pretrained = False
    # Provide a weights path even if pretrained==False
    # This location will be used to store the best weights
    # Just the file_name is enough, automagically stored in export_dir
    weights_path = 'weights.pt'
    
    """ Save trainable train and valid data """
    # Saves trainable data after transforms are applied as HDF5 files
    # This can then be used alongside a simple DataLoader
    # Set batch_size to 1, so each sample is stored separately (unless chunks are req.)
    ## WARNING: This may take a while. Check storage space before continuing.
    save_trainable_dataset = False
    
    """ Epochs and Batches """
    num_steps = 25000
    num_epochs = 25
    # Equal batch_size for both training and validation
    batch_size = 100
    # every 'n' epochs
    save_freq = 5
    # Overfitting check and Early Stopping
    early_stopping = True
    
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
    # This is automatically written in export_dir
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
    # The following transforms are performed in the given order
    transforms = dict(
        train=Unify([
            BandPass(lower=16, upper=512, fs=2048., order=6),
            Whiten(trunc_method='hann', remove_corrupted=True),
            MultirateSampling(),
        ]),
        test=Unify([
            BandPass(lower=16, upper=512, fs=2048., order=6),
            Whiten(trunc_method='hann', remove_corrupted=True),
            MultirateSampling(),
        ]),
        target=None
    )
    
    # Debugging (size: train_data = 1e4, val_data = 1e3)
    debug = False
    # Verbosity (Not properly implemented yet. Use logger.)
    verbose = False
    
    
    
""" CUSTOM MODELS FOR EXPERIMENTATION """

class KaggleFirst:
    
    """ Data storage """
    name = "Baseline_kaggle_test"
    export_dir = Path("/Users/nnarenraju/Desktop") / name
    
    """ Dataset Splitting """
    n_splits = 2
    seed = 42
    splitter = None
    
    """ Dataset """
    dataset = MLMDC1
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel
    
    model_params = dict(
        # Kaggle frontend+backend
        # This model is ridiculously slow on cpu, use cuda:0
        model_name = 'kaggle_first', 
        filter_size = 16,
        kernel_size = 32,
        timm_params = {'model_name': 'resnet34', 
                       'pretrained': True, 
                       'in_chans': 2, 
                       'drop_rate': 0.25},
        store_device = 'cuda:0',
    )
    
    pretrained = False
    weights_path = 'weights.pt'
    
    """ Save trainable train and valid data """
    save_trainable_dataset = False
    
    """ Epochs and Batches """
    num_steps = 25000
    num_epochs = 25
    batch_size = 100
    save_freq = 5
    early_stopping = True
    
    """ Gradient Clipping """
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
    loss_function = regularised_BCELoss(dim=2)
    output_loss_file = "losses.txt"
    
    """ Evaluation Metric """
    eval_metric = None
    accuracy_thresh = 0.5
    
    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'
    
    """ Data Transforms """
    transforms = dict(
        train=Unify([
            BandPass(lower=16, upper=512, fs=2048., order=6),
            Whiten(trunc_method='hann', remove_corrupted=True),
            MultirateSampling(),
        ]),
        test=Unify([
            BandPass(lower=16, upper=512, fs=2048., order=6),
            Whiten(trunc_method='hann', remove_corrupted=True),
            MultirateSampling(),
        ]),
        target=None
    )
    
    debug = False
    verbose = False
