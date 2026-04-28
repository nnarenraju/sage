#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : director.py
Description     : Short description of the file

Created on 2021-11-21 15:30:43

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, Sage
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = University of Glasgow
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: https://github.com/nnarenraju/sage

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
from sage.factory.testing import run_test
from sage.benchmark.mlgwsc1.evaluator import main as evaluator
from sage.factory.legacy import train as manual_train
from sage.exec.data_handler import DataModule as dat


# Tensorboard
from torch.utils.tensorboard import SummaryWriter

# Logging
from sage.core.logger import setup_logging


def parse_args():
    """Argument parser for directors

    Returns:
        opts (namespace object): populated namespace object
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="Baseline",
        help="Uses the pipeline architecture as described in configs.py",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="Default",
        help="Creates or uses a particular dataset as provided in data_configs.py",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Running the inference module using trained model",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Running the pipeline in 1 epoch validate mode",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Store model summary using pytorch_summary",
    )

    opts = parser.parse_args()
    return opts


class SageDirector:

    def __init__(self, opts):
        self.opts = opts
        self.cfg = None
        self.data_cfg = None
        self.Network = None
        self.optimiser = None
        self.scheduler = None
        self.loss_function = None
        self.train_loader = None
        self.val_loader = None
        self.aux_loader = None
        self.checkpoint = None

    def prepare_configs(self):
        # Get data creation/usage configuration
        self.data_cfg = dat.configure_dataset(self.opts)
        # Get model configuration
        self.cfg = dat.configure_pipeline(self.opts)

    def prepare_data(self, train_fold=None, val_fold=None, balance_params=None):
        # Get input data length
        # Used in torch summary and to initialise norm layers
        dat.input_sample_length(self.data_cfg)

        # Make export dir (save dir for all outputs)
        # TODO: Save config file as YAML/JSON
        # TODO: Save a copy of the code files used for the run
        # TODO: Freeze git commit hash and save
        dat.make_export_dir(self.cfg)

        # Prepare input data for training and testing
        # This should create/use a dataset and save a copy of the lookup table
        # This function does nothing if OTF is True
        dat.get_summary(self.cfg, self.data_cfg, self.cfg.export_dir)

        # Get the dataset objects for training and validation
        self.train_data, self.val_data, self.aux_data = dat.get_dataset_objects(
            self.cfg, self.data_cfg, train_fold, val_fold
        )

        # Get the Pytorch DataLoader objects of train and valid data
        self.train_loader, self.val_loader, self.aux_loader, self.nepoch, self.cflag = (
            dat.get_dataloader(
                self.cfg, self.train_data, self.val_data, self.aux_data, balance_params
            )
        )

    def _freeze_layers(self):
        # Freeze all layers
        # Frozen layers should have no grad before and after backward() call
        # Check using print(model.layer.weight.grad) before and after backward
        for param in self.Network.parameters():
            param.requires_grad = False
        # Unfreeze required layers
        # FIX ME!!! Add this to cfg as option
        layer_names = ["signal_or_noise", "chirp_mass", "coalescence_time"]
        layer_params = [getattr(self.Network, foo).parameters() for foo in layer_names]
        for layer in layer_params:
            for param in layer:
                param.requires_grad = True

    def initialise_model(self):
        # Initialise chosen model architecture (Backend + Frontend)
        self.cfg.model_params.update(
            dict(
                _input_length=self.data_cfg.network_sample_length,
                _decimated_bins=self.data_cfg._decimated_bins,
            )
        )
        # Init Network
        self.Network = self.cfg.model(**self.cfg.model_params)

        # Load weights snapshot
        if self.cfg.pretrained and self.cfg.weights_path != "":
            if os.path.exists(self.cfg.weights_path):
                weights = torch.load(self.cfg.weights_path, self.cfg.store_device)
                self.Network.load_state_dict(weights)
                del weights
                gc.collect()
                # summary(Network, (2, 4096), batch_size=cfg.batch_size)
                # Freezing
                if self.cfg.freeze_for_transfer:
                    self._freeze_layers()
            else:
                raise ValueError("train.py: cfg.weights_path does not exist!")

        elif self.cfg.pretrained and self.cfg.weights_path == "":
            raise ValueError("CFG: pretrained==True, but no weights path provided!")

    def prepare_model_summary(self):
        # Model Summary (frontend + backend)
        if self.opts.summary:
            # Using TorchSummary to get # trainable params and general overview
            summary(
                self.Network,
                (2, self.data_cfg.network_sample_length),
                batch_size=self.cfg.batch_size,
            )
            print("")
            # Using TensorBoard summary writer to create detailed graph of ModelClass
            tb = SummaryWriter()
            samples, labels = next(iter(self.train_loader))
            tb.add_graph(self.Network, samples)
            tb.close()

    def prepare_optimiser(self):
        # Optimiser and Scheduler (Set to None if unused)
        if self.cfg.optimiser is not None:
            self.optimiser = self.cfg.optimiser(
                self.Network.parameters(), **self.cfg.optimiser_params
            )
        else:
            self.optimiser = None

    def prepare_scheduler(self):
        if self.cfg.scheduler is not None:
            self.scheduler = self.cfg.scheduler(
                self.optimiser, **self.cfg.scheduler_params
            )
        else:
            self.scheduler = None
        return self.scheduler

    def prepare_loss_function(self):
        # Loss function used
        self.loss_function = self.cfg.loss_function

    def prepare_checkpoint(self):
        # Resume training by loading a checkpoint file or prepare checkpoint
        if self.cfg.resume_from_checkpoint:
            self.checkpoint = torch.load(
                self.cfg.checkpoint_path, map_location=self.cfg.train_device
            )
            self.Network.load_state_dict(self.checkpoint["model_state_dict"])
            self.optimiser.load_state_dict(self.checkpoint["optimiser_state_dict"])

    def train_model(self):
        # Initialise the trainer
        self.Network = manual_train(
            self.cfg,
            self.data_cfg,
            self.train_data,
            self.val_data,
            self.Network,
            self.optimiser,
            self.scheduler,
            self.loss_function,
            self.train_loader,
            self.val_loader,
            self.aux_loader,
            self.nepoch,
            self.cflag,
            self.checkpoint,
            self.opts.validate,
            verbose=self.cfg.verbose,
        )

    def test_model(self):
        if opts.inference:
            # Running the testing phase for foreground data
            transforms = self.cfg.transforms["test"]
            jobs = ["foreground", "background"]

            self.output_testing_dir = os.path.join(self.cfg.export_dir, "TESTING")
            for job in jobs:
                # Get the required data based on testing job
                if job == "foreground":
                    testfile = os.path.join(
                        self.cfg.testing_dir, self.cfg.test_foreground_dataset
                    )
                    evalfile = os.path.join(
                        self.output_testing_dir, self.cfg.test_foreground_output
                    )
                elif job == "background":
                    testfile = os.path.join(
                        self.cfg.testing_dir, self.cfg.test_background_dataset
                    )
                    evalfile = os.path.join(
                        self.output_testing_dir, self.cfg.test_background_output
                    )

                print("\nRunning the testing phase on {} data".format(job))
                run_test(
                    self.Network,
                    testfile,
                    evalfile,
                    transforms,
                    self.cfg,
                    self.data_cfg,
                    step_size=self.cfg.step_size,
                    slice_length=int(
                        self.data_cfg.signal_length * self.data_cfg.sample_rate
                    ),
                    trigger_threshold=self.cfg.trigger_threshold,
                    cluster_threshold=self.cfg.cluster_threshold,
                    batch_size=self.cfg.batch_size,
                    device=self.cfg.testing_device,
                    verbose=self.cfg.verbose,
                )

    def evaluate_model(self):
        # Run the evaluator for the testing phase and add required files to TESTING dir in export_dir
        raw_args = [
            "--injection-file",
            os.path.join(self.cfg.testing_dir, self.cfg.injection_file),
        ]
        raw_args += [
            "--foreground-events",
            os.path.join(self.output_testing_dir, self.cfg.test_foreground_output),
        ]
        raw_args += [
            "--foreground-files",
            os.path.join(self.cfg.testing_dir, self.cfg.test_foreground_dataset),
        ]
        raw_args += [
            "--background-events",
            os.path.join(self.output_testing_dir, self.cfg.test_background_output),
        ]
        out_eval = os.path.join(self.output_testing_dir, self.cfg.evaluation_output)
        raw_args += ["--output-file", out_eval]
        raw_args += ["--output-dir", self.output_testing_dir]
        raw_args += ["--verbose"]

        # Running the evaluator to obtain output triggers (with clustering)
        evaluator(
            raw_args,
            cfg_far_scaling_factor=float(self.cfg.far_scaling_factor),
            dataset=self.data_cfg.dataset,
        )

    def save_results(self):
        pass

    def run(self):
        self.prepare_configs()
        self.prepare_data()
        self.initialise_model()
        self.prepare_model_summary()
        self.prepare_optimiser()
        self.prepare_scheduler()
        self.prepare_loss_function()
        self.prepare_checkpoint()
        self.train_model()
        # self.test_model()
        # self.evaluate_model()
        self.save_results()


if __name__ == "__main__":

    # Initialising logger
    # sets up main log and console output
    setup_logging("logs")

    # Parse command-line arguments
    opts = parse_args()

    # Run the director
    director = SageDirector(opts)
    director.run()

    # That's all folks!
    print("\nFIN")
