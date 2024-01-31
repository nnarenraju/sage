# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Sun Nov 21 15:30:43 2021

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
import os
import gc
import torch
import argparse

from functools import partial
from torchsummary import summary

# Warnings
import warnings
warnings.filterwarnings("ignore")

# LOCAL
from test import run_test
from save_online import save
from evaluate import main as evaluator
from manual import train as manual_train
from data.prepare_data import DataModule as dat

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

# RayTune
from ray import tune


def trainer(rtune=None, checkpoint_dir=None, args=None):
    
    cfg, data_cfg, opts, train_fold, val_fold, balance_params = args
    # Get the dataset objects for training and validation
    train_data, val_data = dat.get_dataset_objects(cfg, data_cfg, train_fold, val_fold)
    
    # Get the Pytorch DataLoader objects of train and valid data
    train_loader, val_loader, nepoch = dat.get_dataloader(cfg, train_data, val_data, balance_params)
    
    # Initialise chosen model architecture (Backend + Frontend)
    # Equivalent to the "Network" variable in manual mode without weights
    cfg.model_params.update(dict(_input_length=data_cfg.network_sample_length,
                                 _decimated_bins=data_cfg._decimated_bins))
    # Init Network
    Network = cfg.model(**cfg.model_params)
    
    # Load weights snapshot
    if cfg.pretrained and cfg.weights_path!='':
        if os.path.exists(cfg.weights_path):
            weights = torch.load(cfg.weights_path, cfg.store_device)
            Network.load_state_dict(weights)
            del weights; gc.collect()
        else:
            raise ValueError("train.py: cfg.weights_path does not exist!")
    elif cfg.pretrained and cfg.weights_path=='':
        raise ValueError("CFG: pretrained==True, but no weights path provided!")
    
    ## Display
    print("Sample length for training and testing = {}".format(data_cfg.network_sample_length))

    # Model Summary (frontend + backend)
    if opts.summary:
        # Using TorchSummary to get # trainable params and general overview
        summary(Network, (2, data_cfg.network_sample_length), batch_size=cfg.batch_size)
        print("")
        # Using TensorBoard summary writer to create detailed graph of ModelClass
        tb = SummaryWriter()
        samples, labels = next(iter(train_loader))
        tb.add_graph(Network, samples)
        tb.close()
    
    # Optimizer and Scheduler (Set to None if unused)
    if cfg.optimizer is not None:
        optimizer = cfg.optimizer(Network.parameters(), **cfg.optimizer_params)
    else:
        optimizer = None
        
    if cfg.scheduler is not None:
        scheduler = cfg.scheduler(optimizer, **cfg.scheduler_params)
    else:
        scheduler = None
    
    # Loss function used
    loss_function = cfg.loss_function
    
    """ RayTune Checkpointing """
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        Network.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    """ Training and Validation Methods """
    ## MANUAL
    if opts.manual:
        # Running the manual pipeline version using pure PyTorch
        # Initialise the trainer
        Network = manual_train(cfg, data_cfg, train_data, val_data, Network, optimizer, scheduler, loss_function,
                               train_loader, val_loader, nepoch, verbose=cfg.verbose)
    
    if rtune == None:
        return Network


def run_trainer():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, default='Baseline',
                        help="Uses the pipeline architecture as described in configs.py")
    parser.add_argument("--data-config", type=str, default='Default',
                        help="Creates or uses a particular dataset as provided in data_configs.py")
    parser.add_argument("--inference", action='store_true',
                        help="Running the inference module using trained model")
    parser.add_argument("--manual", action='store_true',
                        help="Running the pipeline using manual PyTorch")
    parser.add_argument("--summary", action='store_true',
                        help="Store model summary using pytorch_summary")
    
    opts = parser.parse_args()
    
    
    """ Prepare Data """
    # Get model configuration
    cfg = dat.configure_pipeline(opts)
    # Get data creation/usage configuration
    data_cfg = dat.configure_dataset(opts)

    # Get input data length
    # Used in torch summary and to initialise norm layers
    dat.input_sample_length(data_cfg)
    
    # Make export dir
    dat.make_export_dir(cfg)
    
    # Prepare input data for training and testing
    # This should create/use a dataset and save a copy of the lookup table
    dat.get_summary(cfg, data_cfg, cfg.export_dir)
    
    if not data_cfg.OTF:
        # Prepare dataset (read, split and return fold idx)
        # Folds are based on stratified-KFold method in Sklearn (preserves class ratio)
        train, folds, balance_params = dat.get_metadata(cfg, data_cfg)

        """ Training (non-OTF) """
        # Folds are obtained only by splitting the training dataset
        # Use folds for cross-validation
        for nfold, (train_idx, val_idx) in enumerate(folds):
            
            if cfg.splitter != None:
                raise NotImplementedError('K-fold cross validation method under construction!')
                print(f'\n========================= TRAINING FOLD {nfold} =========================\n')

            train_fold = train.iloc[train_idx]
            val_fold = train.iloc[val_idx]
            
            if cfg.rtune_optimise:
                ### RayTune needs to optimise parameter from this point forward
                ### All code before inference section must be wrapped into a train function
                rtune = cfg.rtune_params
                # Get RayTune configs
                rtune_config = rtune['config']
                rtune_scheduler = rtune['scheduler'](**rtune['scheduler_params'])
                rtune_reporter = rtune['reporter'](**rtune['reporter_params'])
                
                # Constant args to be passed to trainer
                const = (cfg, data_cfg, opts, train_fold, val_fold, balance_params, )
                
                # Run RayTune with given config options
                result = tune.run(
                    partial(trainer, args=const),
                    name = "raytune_optimisation",
                    local_dir = os.path.join(cfg.export_dir, "raytune"),
                    resources_per_trial = {"cpu": 1, "gpu": 1},
                    max_concurrent_trials=1,
                    config = rtune_config,
                    num_samples = rtune['num_samples'],
                    scheduler = rtune_scheduler,
                    progress_reporter = rtune_reporter
                )
                
                # Getting the best trial
                # Syntax: get_best_trial(metric, mode, scope)
                # Usage: esult.get_best_trial("loss", "min", "last")
                # Meaning: Get the trial that has minimum validation loss
                best_trial = result.get_best_trial("loss", "min", "last")
                print("Best trial config: {}".format(best_trial.config))
                print("Best trial final validation loss: {}".format(
                    best_trial.last_result["loss"]))
                
                # Best trained Network
                best_checkpoint_dir = best_trial.checkpoint.value
                model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
                # Loading the network the best weights
                Network = cfg.model(**cfg.model_params)
                Network.load_state_dict(model_state)
                
                ### RayTune Optimisation ends ###
            
            else:
                # Running the pipeline without RayTune optimisation
                args = (cfg, data_cfg, opts, train_fold, val_fold, balance_params, )
                Network = trainer(args=args)
            
            # Save run in online workspace
            # save(cfg, data_cfg)
            
            # Sanity check for Network load
            if Network == None:
                return
    
    else:
        args = (cfg, data_cfg, opts, None, None, None, )
        Network = trainer(args=args)
        
    """ TESTING """
    if opts.inference:
        # Running the testing phase for foreground data
        transforms = cfg.transforms['test']
        jobs = ['foreground', 'background']
        
        output_testing_dir = os.path.join(cfg.export_dir, 'TESTING')
        for job in jobs:
            # Get the required data based on testing job
            if job == 'foreground':
                testfile = os.path.join(cfg.testing_dir, cfg.test_foreground_dataset)
                evalfile = os.path.join(output_testing_dir, cfg.test_foreground_output)
            elif job == 'background':
                testfile = os.path.join(cfg.testing_dir, cfg.test_background_dataset)
                evalfile = os.path.join(output_testing_dir, cfg.test_background_output)
                
            print('\nRunning the testing phase on {} data'.format(job))
            run_test(Network, testfile, evalfile, transforms, cfg, data_cfg,
                        step_size=cfg.step_size, slice_length=int(data_cfg.signal_length*data_cfg.sample_rate),
                        trigger_threshold=cfg.trigger_threshold, cluster_threshold=cfg.cluster_threshold, 
                        batch_size = cfg.batch_size,
                        device=cfg.testing_device, verbose=cfg.verbose)
    
        # Run the evaluator for the testing phase and add required files to TESTING dir in export_dir
        raw_args =  ['--injection-file', os.path.join(cfg.testing_dir, cfg.injection_file)]
        raw_args += ['--foreground-events', os.path.join(output_testing_dir, cfg.test_foreground_output)]
        raw_args += ['--foreground-files', os.path.join(cfg.testing_dir, cfg.test_foreground_dataset)]
        raw_args += ['--background-events', os.path.join(output_testing_dir, cfg.test_background_output)]
        out_eval = os.path.join(output_testing_dir, cfg.evaluation_output)
        raw_args += ['--output-file', out_eval]
        raw_args += ['--output-dir', output_testing_dir]
        raw_args += ['--verbose']
        
        # Running the evaluator to obtain output triggers (with clustering)
        evaluator(raw_args, cfg_far_scaling_factor=float(cfg.far_scaling_factor), dataset=data_cfg.dataset)

        

if __name__ == "__main__":
    
    run_trainer()
    print('\nFIN')    
