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
import json
import math
import torch
from pathlib import Path
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# LOCAL
from data.datasets import MLMDC1_IterSample, BatchLoader
from architectures.frontend import GammaModel, KappaModel
from data.transforms import Unify, UnifySignal, UnifyNoise
from data.transforms import BandPass, HighPass, Whiten, MultirateSampling
from data.transforms import AugmentDistance, AugmentPolSky, CyclicShift
from losses.custom_loss_functions import BCEgw_MSEtc, regularised_BCELoss, regularised_BCEWithLogitsLoss


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
    dataset = MLMDC1_IterSample
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
    # MegaBatch Method (Datasets are produced in large chunk files)
    megabatch = False
    # Overfitting check and Early Stopping
    early_stopping = True
    
    """ DataLoader params """
    num_workers = 0
    pin_memory = False
    
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
    debug_size = 10000
    # Verbosity (Not properly implemented yet. Use logger.)
    verbose = False
    
    
    
""" CUSTOM MODELS FOR EXPERIMENTATION """

class KaggleFirst:
    
    """ Data storage """
    name = "KaggleFirst"
    export_dir = Path("/Users/nnarenraju/Desktop") / name
    online_workspace = "/data/www.astro/nnarenraju"
    save_remarks = ''
    
    """ Dataset Splitting """
    n_splits = 2
    seed = 42
    splitter = None
    
    """ Dataset """
    dataset = MLMDC1_IterSample
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
        store_device = 'cpu',
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
    megabatch = False
    early_stopping = False
    
    """ Dataloader params """
    num_workers = 0
    pin_memory = False
    prefetch_factor = 2
    persistent_workers = False
    
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
    store_device = 'cpu'
    train_device = 'cpu'
    
    """ Data Transforms """
    # Adding a random noise realisation during the data loading process
    # Procedure should be available within dataset object
    add_random_noise_realisation = True
    
    transforms = dict(
        signal=None,
        noise=None,
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
    debug_size = 10000
    
    verbose = False


class KF_Trainable(KaggleFirst):
    
    """ Parameters changed when creating Batched trainable dataset """
    
    """ Data storage """
    name = "Batch_1"
    export_dir = Path("/Users/nnarenraju/Desktop") / name
    
    """ Dataset Splitting """
    # Split the iterable into several small iterables
    est_file_size = 0.7 # MB
    # Assumed to load entire batch into memory (if needed)
    limit_RAM = 8000. # MB
    # Number of files in each split
    fold_size = limit_RAM/est_file_size
    # Total number of input files/ number of samples
    num_samples = 100
    # Number of splits used to create Stratified folds
    # Number of folds (must be at least 2, default = 5)
    n_splits = int(math.ceil(num_samples/fold_size))
    
    # Seed for K-Fold shuffling
    seed = 42
    # Folds are made by preserving the percentage of samples for each class (StratifiedKFold)
    # Each of these folds are stored as a batch of data
    # Uses the splitter.split method with samples and target as input
    # splitter = StratifiedKFold(n_splits=n_splits)
    
    # If splitter is set to None, we store the entire data into one HDF
    splitter = None
    
    # Saves trainable transformed dataset
    # Pipeline does not use the dataset object within datasets.py
    # It has a custom version within save_trainable.py
    # TODO: this can eventually be moved to datasets.py as well
    save_trainable_dataset = True
    
    
class KF_BatchTrain(KaggleFirst):
    
    """ Parameters changed when creating Batched trainable dataset """
    
    """ Data storage """
    name = "KF_D1_BatchTrain"
    export_dir = Path("/Users/nnarenraju/Desktop") / name
    
    """ Dataset """
    dataset = BatchLoader
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
        store_device = 'cpu',
    )
    
    pretrained = False
    weights_path = 'weights.pt'
    
    """ Epochs and Batches """
    num_steps = 25000
    num_epochs = 25
    batch_size = 100
    save_freq = 5
    early_stopping = False
    
    """ Data Transforms """
    # Adding a random noise realisation during the data loading process
    # Procedure should be available within dataset object
    add_random_noise_realisation = True
    
    """ Loss Function """
    loss_function = BCEgw_MSEtc()
    output_loss_file = "losses.txt"
    
    transforms = dict(
        signal=UnifySignal([
            AugmentPolSky(),
            AugmentDistance(),
        ]),
        noise=UnifyNoise([
            CyclicShift(),
        ]),
        train=Unify([
            HighPass(lower=16, fs=2048., order=6),
            Whiten(trunc_method='hann', remove_corrupted=True),
            MultirateSampling(),
        ]),
        test=Unify([
            HighPass(lower=16, fs=2048., order=6),
            Whiten(trunc_method='hann', remove_corrupted=True),
            MultirateSampling(),
        ]),
        target=None
    )
    
    """ Storage Devices """
    store_device = 'cpu'
    train_device = 'cpu'
    
    debug = False


class Baseline_May18(KF_BatchTrain):
    
    """ Data storage """
    name = "Baseline_May18"
    export_dir = Path("/Users/nnarenraju/Desktop") / name
    
    """ Dataset """
    dataset = MLMDC1_IterSample
    dataset_params = dict()
    
    """ Architecture """
    model = GammaModel
    
    model_params = dict(
        # Simple NN Model
        model_name = 'mlmdc_example',
        in_channels = 2,
        out_channels = 1,
        store_device = 'cpu',
    )
    
    """ Dataloader params """
    num_workers = 8
    pin_memory = True
    prefetch_factor = 10
    persistent_workers = True
    
    """  DataLoader mode """
    megabatch = False
    
    """ Loss Function """
    loss_function = regularised_BCELoss(dim=1)


class KaggleFirst_Jun9(KF_BatchTrain):
    
    """ Data storage """
    name = "KaggleFirst_Jul8"
    export_dir = Path("/home/nnarenraju/Research") / name
    save_remarks = ''
    
    """ Dataset """
    dataset = MLMDC1_IterSample
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel
    
    model_params = dict(
        # Kaggle frontend+backend
        # This model is ridiculously slow on cpu, use cuda:0
        model_name = 'kaggle_first', 
        filter_size = 32,
        kernel_size = 64,
        timm_params = {'model_name': 'resnet34', 
                        'pretrained': True, 
                        'in_chans': 2, 
                        'drop_rate': 0.25},
        store_device = 'cuda:1',
    )
    
    """ Scheduler """
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    
    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 100
    persistent_workers = True
    
    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'
    
    """  DataLoader mode """
    megabatch = False
    
    """ Loss Function """
    loss_function = regularised_BCEWithLogitsLoss(dim=1)
    
    """ Optimizer """
    # optimizer = optim.SGD
    # optimizer_params = dict(lr=2e-4, momentum=0.9)
    
    debug = True
    debug_size = 1000
    
    



""" Load config file into class """

class Imported:
    
    def __init__(self, imported_config):
        with open(imported_config) as fp:
            cfg = json.load(fp)
        
        """ Data storage """
        self.name = cfg['name']
        self.export_dir = Path(cfg['export_dir'])
        
        """ Dataset Splitting """
        self.n_splits = 2
        self.seed = 42
        self.splitter = None
        
        """ Dataset """
        self.dataset = MLMDC1_IterSample
        self.dataset_params = dict()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
