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

Documentation:

[1] Using OSnet

##Architecture
model = SigmaModel

# Kernel sizes on modified OSnet (type 1)
kernel_sizes = []
kernel_sizes.append([[16, 32, 64, 128, 256], [8, 16, 32, 64, 128]])
kernel_sizes.append([[8, 16, 32, 64, 128], [2, 4, 8, 16, 32]])
kernel_sizes.append([[2, 4, 8, 16, 32], [2, 4, 8, 16, 32]])

model_params = dict(
    ## OSnet + Resnet50 CBAM
    model_name='sigmanet',
    norm_layer = 'instancenorm',
    ## OSnet params
    # channels[0] is used when initial_dim_reduction == True
    channels=[16, 32, 64, 128],
    kernel_sizes=kernel_sizes, 
    # strides[:2] is used when initial_dim_reduction == True
    strides=[2,2,8,4],
    stacking=False,
    initial_dim_reduction=False,
    # reduction value of 16 does not work with KaggleNet type kernels
    channel_gate_reduction=8,
    # ResNet CBAM params
    resnet_size = 50,
    # Common
    store_device = 'cuda:2',
)

"""

# PACKAGES
import os
import torch
import subprocess
import numpy as np
import torch.optim as optim

from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

## LOCAL
# Dataset objects
from data.datasets import MinimalOTF

# Architectures
from architectures.models import Rigatoni_MS_ResNetCBAM, Rigatoni_MS_ResNetCBAM_legacy
from architectures.models import KappaModel_ResNet1D
from architectures.frontend import MultiScaleBlock

# Transforms, augmentation and generation
from data.transforms import Unify, UnifySignal, UnifyNoise, UnifySignalGen, UnifyNoiseGen
from data.transforms import Whiten, MultirateSampling, Normalise
from data.transforms import AugmentOptimalNetworkSNR, AugmentPolSky
from data.transforms import Recolour
# Generating signals and noise
from data.transforms import FastGenerateWaveform, SinusoidGenerator
from data.transforms import RandomNoiseSlice, MultipleFileRandomNoiseSlice, ColouredNoiseGenerator

# Loss functions
from losses.custom_loss_functions import BCEWithPEregLoss

# RayTune
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# TASKS
# Cleanup prior modifications
# Change all mention on ORChiD to Sage
# All tmp files must be placed in a single location (eg. segments.csv)
# code to produce all tmp files must be consolidated (add to utils)
# code to download noise files for full experimentation must be consolidated (add to utils)
# Change all debug folders to exist within export_dir
# Clean unify noise gen
# Move all external data into one directory (psds, O3 noise, etc.)
# Add verbosity to all modules

# Add logging to all modules
# Add documentation to all classes and functions
# Add diagnostic tests with at least 90% coverage
# Add Sage logo to output

    
""" CUSTOM MODELS FOR EXPERIMENTATION """

class SageNetOTF:
    
    """ Data storage """
    name = "SageNet50_CBAM_OTF_Feb03_dummy"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')

    """ RayTune (Untested) """
    # Placed before initialising any relevant tunable parameter
    # WARNING: Required compute is prohibitively large for large models
    rtune_optimise = False
    
    rtune_params = dict(
        # RayTune Tunable Parameters
        config = {
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([32,])
        },
        # Scheduler (ASHA has intelligent early stopping)
        scheduler = ASHAScheduler,
        # NOTE: max_t is maximum number of epochs Tune is allowed to run
        scheduler_params = dict(
            metric = "loss",
            mode = "min",
            max_t = 10,
            grace_period = 1,
            reduction_factor = 2
        ),
        # Reporter
        reporter = CLIReporter,
        reporter_params = dict(
            metric_columns=["loss", "accuracy", "low_far_nsignals", "training_iteration"]
        ),
        # To sample multiple times/run multiple trials
        # Samples from config num_samples number of times
        num_samples = 100
    )

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM

    # Following options available for pe point estimate
    # 'norm_tc', 'norm_dchirp', 'norm_mchirp', 
    # 'norm_dist', 'norm_q', 'norm_invq', 'norm_snr'
    model_params = dict(
        scales = [1, 2, 4, 0.5, 0.25],
        blocks = [
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock]
        ],
        out_channels = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes = [
            [64, 64 // 2 + 1], 
            [64 // 2 + 1, 64 // 4 + 1], 
            [64 // 4 + 1, 64 // 4 + 1]
        ], 
        compression_factor = [8, 4, 0],
        in_channels = 1,
        resnet_size = 50,
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
        norm_layer = 'instancenorm',
        store_device = 'cuda:0',
        review = True
    )

    """ Epochs and Batches """
    num_epochs = 500
    batch_size = 64
    validation_plot_freq = 1 # every n epochs
    
    """ Weight Types """
    # Lowest loss, highest accuracy, highest auc, highest low_far_nsignals found
    weight_types = ['loss', 'accuracy', 'roc_auc', 'low_far_nsignals']
    
    # Save weights for particular epochs
    save_epoch_weight = list(range(500))
    
    # Pick one of the above weights for best epoch save directory
    save_best_option = 'loss'

    # Checkpoints
    save_checkpoint = True
    checkpoint_freq = 1 # every n epochs
    resume_from_checkpoint = False
    checkpoint_path = ""
    
    pretrained = False
    freeze_for_transfer = False
    weights_path = 'weights_loss.pt'

    """ Optimizer """
    ## Adam 
    optimizer = optim.Adam
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)

    """ Scheduler """
    ## Cosine Annealing with Warm Restarts
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    
    """ Gradient Clipping """
    clip_norm = 10000

    """ Automatic Mixed Precision """
    # Keep this turned off when using Adam
    # It seems to be unstable and produces NaN losses
    do_AMP = False

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    """ Dataloader params """
    num_workers = 32
    pin_memory = True
    prefetch_factor = 4
    persistent_workers = True

    """ Loss Function """
    # All parameter estimation is done only using MSE loss at the moment
    loss_function = BCEWithPEregLoss(gw_loss=torch.nn.BCEWithLogitsLoss(), mse_alpha=1.0)
    
    # Calculate the network SNR for pure noise samples as well
    # If used with parameter estimation, custom loss function should have network_snr_for_noise option toggled
    network_snr_for_noise = False

    # Dataset imbalance
    ignore_dset_imbalance = False
    subset_for_funsies = False # debug_size is used for subset, debug need not be true

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False
                                ),
                    },
                    # Auxilliary noise data (only used for training, not for validation)
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )
    
    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_30days.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_30days.hdf"),
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
                ]),
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )

    """ Optional things to do during training """
    # Testing on a small 64000s dataset at the end of each epoch
    epoch_testing = False
    epoch_testing_dir = "/local/scratch/igr/nnarenraju/testing_64000_D4_seeded"
    epoch_far_scaling_factor = 64000.0

    """ Testing Phase """
    injection_file = 'injections.hdf'
    evaluation_output = 'evaluation.hdf'

    test_foreground_dataset = "foreground.hdf"
    test_foreground_output = "testing_foutput.hdf"
    
    test_background_dataset = "background.hdf"
    test_background_output = "testing_boutput.hdf"

    # Run device for testing phase
    ## Testing config
    # Real step will be slightly different due to rounding errors
    step_size = 0.1
    # Based on prediction probabilities in best epoch
    trigger_threshold = 0.0
    # Time shift the signal by multiple of step_size and check pred probs
    cluster_threshold = 0.0001
    # Run device for testing phase
    testing_device = 'cuda:1'

    testing_dir = "/local/scratch/igr/nnarenraju/testing_month_D4_seeded"
    far_scaling_factor = 2592000.0

    # Debugging
    debug = False
    debug_size = 10000

    verbose = True


# No rerun planned (will not be a part of bug fixes)
class Yukon_Feb24(SageNetOTF):
    """ Data storage """
    name = "SageNet50_CBAM_OTF_Feb03_Yukon"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM

    # Following options available for pe point estimate
    # 'norm_tc', 'norm_dchirp', 'norm_mchirp', 
    # 'norm_dist', 'norm_q', 'norm_invq', 'norm_snr'
    model_params = dict(
        scales = [1, 2, 4, 0.5, 0.25],
        blocks = [
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock]
        ],
        out_channels = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes = [
            [64, 64 // 2 + 1], 
            [64 // 2 + 1, 64 // 4 + 1], 
            [64 // 4 + 1, 64 // 4 + 1]
        ], 
        compression_factor = [8, 4, 0],
        in_channels = 1,
        resnet_size = 50,
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
        norm_layer = 'instancenorm',
        store_device = 'cuda:0',
        review = False
    )

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'


# No rerun planned (will not be a part of bug fixes)
class Yukon_Feb24_rerun(SageNetOTF):
    """ Data storage """
    # Apr21 IMRPhenomD completed at 50 epochs. Rerun with checkpoint file from last epoch.
    name = "SageNet50_CBAM_IMRPhenomD_OTF_May01_Yukon_rerun"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM

    # Following options available for pe point estimate
    # 'norm_tc', 'norm_dchirp', 'norm_mchirp', 
    # 'norm_dist', 'norm_q', 'norm_invq', 'norm_snr'
    model_params = dict(
        scales = [1, 2, 4, 0.5, 0.25],
        blocks = [
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock]
        ],
        out_channels = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes = [
            [64, 64 // 2 + 1], 
            [64 // 2 + 1, 64 // 4 + 1], 
            [64 // 4 + 1, 64 // 4 + 1]
        ], 
        compression_factor = [8, 4, 0],
        in_channels = 1,
        resnet_size = 50,
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
        norm_layer = 'instancenorm',
        store_device = 'cuda:0',
        review = False
    )

    """ Checkpoints """
    save_checkpoint = True
    checkpoint_freq = 1 # every n epochs
    resume_from_checkpoint = True
    checkpoint_path = "./weights/checkpoint_epoch_49_PhenomD_BOY.pt"

    """ Epochs and Batches """
    num_epochs = 500
    # Save weights for particular epochs
    save_epoch_weight = list(range(50, 500))

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'


### FINAL EXPERIMENTS ###

# DONE (BEST 24/06/24)
class SageNetOTF_May24_Russet(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)

    """ Data storage """
    name = "SageNet50_halfnormSNR_May17_Russet"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    pretrained = False
    weights_path = "/home/nnarenraju/Research/ORChiD/RUNS/SageNet50_halfnormSNR_May17_Russet/checkpoint_epoch_39.pt"

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_30days.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_30days.hdf"),
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
                ]),
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM_legacy

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:0',
    )
    
    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_BEST_June.hdf"    
    test_background_output = "testing_boutput_BEST_June.hdf"


# DONE (Unbiased and bad noise rejection)
class SageNetOTF_metric_density_Desiree(SageNetOTF):
    ### Primary Deviations (Comparison to BOY latest) ###
    # 1. 113 days of O3b data (not variation)
    # 2. SNR halfnorm (not variation)
    # 3. Template placement density - fixed (**VARIATION**)

    """ Data storage """
    name = "SageNet50_metric_density_Jun26"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_30days.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_30days.hdf"),
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
                ]),
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM

    # Following options available for pe point estimate
    # 'norm_tc', 'norm_dchirp', 'norm_mchirp', 
    # 'norm_dist', 'norm_q', 'norm_invq', 'norm_snr'
    model_params = dict(
        scales = [1, 2, 4, 0.5, 0.25],
        blocks = [
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock]
        ],
        out_channels = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes = [
            [64, 64 // 2 + 1], 
            [64 // 2 + 1, 64 // 4 + 1], 
            [64 // 4 + 1, 64 // 4 + 1]
        ], 
        compression_factor = [8, 4, 0],
        in_channels = 1,
        resnet_size = 50,
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
        norm_layer = 'instancenorm',
        store_device = 'cuda:0',
        review = False
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'
    
    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_metric_latest.hdf"    
    test_background_output = "testing_boutput_metric_latest.hdf"


# DONE (Unbiased lower sensitivity)
class SageNetOTF_metric_density_noCheatyPSDaug_Desiree(SageNetOTF):
    ### Primary Deviations (Comparison to BOY latest) ###
    # 1. 113 days of O3b data (not variation)
    # 2. SNR halfnorm (not variation)
    # 3. Template placement density - fixed (**VARIATION**)

    """ Data storage """
    name = "SageNet50_metric_density_noCheatyPSDaug_Jun26"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             use_shifted=True, shift_up_factor=10, shift_down_factor=1,
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_latter51days_20s.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_latter51days_20s.hdf"),
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
                ]),
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM

    # Following options available for pe point estimate
    # 'norm_tc', 'norm_dchirp', 'norm_mchirp', 
    # 'norm_dist', 'norm_q', 'norm_invq', 'norm_snr'
    model_params = dict(
        scales = [1, 2, 4, 0.5, 0.25],
        blocks = [
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock]
        ],
        out_channels = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes = [
            [64, 64 // 2 + 1], 
            [64 // 2 + 1, 64 // 4 + 1], 
            [64 // 4 + 1, 64 // 4 + 1]
        ], 
        compression_factor = [8, 4, 0],
        in_channels = 1,
        resnet_size = 50,
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
        norm_layer = 'instancenorm',
        store_device = 'cuda:0',
        review = False
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_noCheaty.hdf"    
    test_background_output = "testing_boutput_noCheaty.hdf"


# DONE
class SageNetOTF_metric_density_noCheatyPSDaug_noPSDshift_Desiree(SageNetOTF):
    ### Primary Deviations (Comparison to BOY latest) ###
    # 1. 113 days of O3b data (not variation)
    # 2. SNR halfnorm (not variation)
    # 3. Template placement density - fixed (**VARIATION**)

    """ Data storage """
    name = "SageNet50_metric_density_noCheatyPSDaug_noPSDshift_Jun26"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             use_shifted=False, shift_up_factor=10, shift_down_factor=1,
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_latter51days_20s.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_latter51days_20s.hdf"),
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
                ]),
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM

    # Following options available for pe point estimate
    # 'norm_tc', 'norm_dchirp', 'norm_mchirp', 
    # 'norm_dist', 'norm_q', 'norm_invq', 'norm_snr'
    model_params = dict(
        scales = [1, 2, 4, 0.5, 0.25],
        blocks = [
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock]
        ],
        out_channels = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes = [
            [64, 64 // 2 + 1], 
            [64 // 2 + 1, 64 // 4 + 1], 
            [64 // 4 + 1, 64 // 4 + 1]
        ], 
        compression_factor = [8, 4, 0],
        in_channels = 1,
        resnet_size = 50,
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
        norm_layer = 'instancenorm',
        store_device = 'cuda:0',
        review = False
    )

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_noCheaty_noShift.hdf"    
    test_background_output = "testing_boutput_noCheaty_noShift.hdf"


# RUNNING (Frozen Russet to Desiree)
class Russet_to_Desiree_Annealed(SageNetOTF):
    # Freezing Russet BEST except embedding layer
    # Transfer-learning using Desiree metric density
    # Experiment: Noise rejection from Russet and unbiased from Desiree
    # Russet has long duration signals, so earlier layers of frozen net should be fine

    """ Data storage """
    name = "Russet_to_Desiree_Annealed_July23"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    pretrained = True
    freeze_for_transfer = True
    # weights_path = "/home/nnarenraju/Research/ORChiD/RUNS/transfer/weights_Russet_BEST_epoch_39.pt"
    weights_path = 'weights_loss.pt'

    """ Optimizer """
    ## Adam 
    optimizer = optim.Adam
    optimizer_params = dict(lr=1e-5, weight_decay=1e-6)

    """ Scheduler """
    ## Cosine Annealing with Warm Restarts
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_30days.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_30days.hdf"),
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
                ]),
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM_legacy

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:0',
    )
    
    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_Annealed_Russet_to_Desiree.hdf"
    test_background_output = "testing_boutput_Annealed_Russet_to_Desiree.hdf"


# Anneal from U(m1, m2) to template placement metric (RUNNING)
class Kennebec_Annealed(SageNetOTF):
    # Hopefully we can keep both noise rejection capabilities and unbiased representation

    """ Data storage """
    name = "Kennebec_Annealed_training_Jul25"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    # Weights for testing
    weights_path = 'CHECKPOINTS/checkpoint_epoch_39.pt'

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_30days.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_30days.hdf"),
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
                ]),
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM

    # Following options available for pe point estimate
    # 'norm_tc', 'norm_dchirp', 'norm_mchirp', 
    # 'norm_dist', 'norm_q', 'norm_invq', 'norm_snr'
    model_params = dict(
        scales = [1, 2, 4, 0.5, 0.25],
        blocks = [
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock]
        ],
        out_channels = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes = [
            [64, 64 // 2 + 1], 
            [64 // 2 + 1, 64 // 4 + 1], 
            [64 // 4 + 1, 64 // 4 + 1]
        ], 
        compression_factor = [8, 4, 0],
        in_channels = 1,
        resnet_size = 50,
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
        norm_layer = 'instancenorm',
        store_device = 'cuda:0',
        review = False
    )

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'
    
    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_annealed_training.hdf"    
    test_background_output = "testing_boutput_annealed_training.hdf"


# RUNNING
class Norland_D3_template_density(SageNetOTF):
    # Running D3 on template placement metric
    # Due to the abscence of blip glitches sensitivitiy should not suffer
    # PSD distribution should match exactly between train and test

    """ Data storage """
    name = "Norland_D3_template_placement_metric_Jul26"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': ColouredNoiseGenerator(psds_dir=os.path.join(repo_abspath, "data/psds")),
                    'validation': ColouredNoiseGenerator(psds_dir=os.path.join(repo_abspath, "data/psds")),
                    },
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=None,
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM

    # Following options available for pe point estimate
    # 'norm_tc', 'norm_dchirp', 'norm_mchirp', 
    # 'norm_dist', 'norm_q', 'norm_invq', 'norm_snr'
    model_params = dict(
        scales = [1, 2, 4, 0.5, 0.25],
        blocks = [
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock]
        ],
        out_channels = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes = [
            [64, 64 // 2 + 1], 
            [64 // 2 + 1, 64 // 4 + 1], 
            [64 // 4 + 1, 64 // 4 + 1]
        ], 
        compression_factor = [8, 4, 0],
        in_channels = 1,
        resnet_size = 50,
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
        norm_layer = 'instancenorm',
        store_device = 'cuda:2',
        review = False
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'
    
    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d3"
    test_foreground_output = "testing_foutput_D3_SageNet.hdf"    
    test_background_output = "testing_boutput_D3_SageNet.hdf"



### PAPER RUNS ###

# RUNNING
class Russet_TrainingRecolour(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)
    # 3. Recoloured using training data (51 days)

    """ Data storage """
    name = "Russet_TrainingRecolour_Aug09"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_latter51days_20s.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_latter51days_20s.hdf"),
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
                ]),
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM_legacy

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:2',
        parameter_estimation = ('norm_tc', 'norm_mchirp', )
    )
    
    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_training_recolour_Aug09.hdf"
    test_background_output = "testing_boutput_training_recolour_Aug09.hdf"


# ABLATION - fixed dataset (RUNNNG)
class Vitelotte_FixedDataset(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)
    # 3. Fixed dataset

    ## What is a fixed dataset?
    # 1. Sample seed does not change between epochs (DONE)
    # 2. Augmentation for signals is the same between epochs (DONE)
    # 3. No augmentation for noise (DONE)

    """ Data storage """
    name = "Vitelotte_FixedDataset_Aug24"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    batchshuffle_noise = True
    
    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentPolSky(),
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=None,
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM_legacy

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
        parameter_estimation = ('norm_tc', 'norm_mchirp', )
    )
    
    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_fixed_dataset.hdf"
    test_background_output = "testing_boutput_fixed_dataset.hdf"


# ABLATION - All the noise, limited signals ()
class Vitelotte_FixedDataset_Relax1(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)
    # 3. Fixed dataset
    # 4. All the noise, limited signals

    """ Data storage """
    name = "Vitelotte_FixedDataset_Aug25_AlltheNoise"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    batchshuffle_noise = False
    
    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = None,
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentPolSky(),
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             use_shifted=False, shift_up_factor=10, shift_down_factor=1,
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_30days_20s.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_30days_20s.hdf"),
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
                ]),
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )

    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM_legacy

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:0',
        parameter_estimation = ('norm_tc', 'norm_mchirp', )
    )
    
    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_fixed_dataset_allnoise.hdf"
    test_background_output = "testing_boutput_fixed_dataset_allnoise.hdf"


# ABLATION - All the signals, limited noise ()
class Vitelotte_FixedDataset_Relax2(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)
    # 3. Fixed dataset
    # 4. All the signal, limited noise

    """ Data storage """
    name = "Vitelotte_FixedDataset_Aug25_AlltheSignals"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    batchshuffle_noise = True
    
    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = None
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=None,
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )

    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM_legacy

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:2',
        parameter_estimation = ('norm_tc', 'norm_mchirp', )
    )
    
    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_fixed_dataset_allsignals.hdf"
    test_background_output = "testing_boutput_fixed_dataset_allsignals.hdf"


# ABLATION 1 - small network resnet50
class Butterball_ResNet1D(SageNetOTF):
    ### Primary Deviations (Comparison to BOY latest) ###
    # 1. 113 days of O3b data (not variation)
    # 2. SNR halfnorm (not variation)

    """ Data storage """
    name = "Butterball_ResNet1D_Aug11"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 100
    persistent_workers = True

    """ Generation """
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/home/nnarenraju/Research/ORChiD/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/home/nnarenraju/Research/ORChiD/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/home/nnarenraju/Research/ORChiD/O3b_real_noise/H1",
                                                            L1="/home/nnarenraju/Research/ORChiD/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_30days.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_30days.hdf"),
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
                ]),
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )
    
    """ Architecture """
    model = KappaModel_ResNet1D

    model_params = dict(
        # Resnet50
        resnet_size = 50,
        store_device = 'cuda:0',
    )

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'
    
    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_resnet1d.hdf"    
    test_background_output = "testing_boutput_resnet1d.hdf"


# BIASES - Spectral Bias
class Rooster_Aug25_SpectralBias(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)

    """ Data storage """
    name = "Rooster_Aug25_SpectralBias"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    SinusoidGenerator(A=1e-20, 
                                      phi=0.0, 
                                      inject_lower = 4.0,
                                      inject_upper = 5.0,
                                      spectral_bias = True,
                                      fixed_duration = 5.0,
                                      lower_freq = 20.0,
                                      upper_freq = 1024.0, 
                                      duration_bias = False,
                                      fixed_frequency = 100.0,
                                      lower_tau = 0.1,
                                      upper_tau = 5.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/home/nnarenraju/Research/ORChiD/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/home/nnarenraju/Research/ORChiD/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/home/nnarenraju/Research/ORChiD/O3b_real_noise/H1",
                                                            L1="/home/nnarenraju/Research/ORChiD/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_30days.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_30days.hdf"),
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
                ]),
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM_legacy

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
    )
    
    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_spectral_bias.hdf"    
    test_background_output = "testing_boutput_spectral_bias.hdf"


# BIASES - Duration Bias
class Rooster_Aug25_DurationBias(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)

    """ Data storage """
    name = "Rooster_Aug25_DurationBias"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    SinusoidGenerator(A=1e-20, 
                                      phi=0.0, 
                                      inject_lower = 4.0,
                                      inject_upper = 5.0,
                                      spectral_bias = False,
                                      fixed_duration = 5.0,
                                      lower_freq = 20.0,
                                      upper_freq = 1024.0, 
                                      duration_bias = True,
                                      fixed_frequency = 100.0,
                                      lower_tau = 0.1,
                                      upper_tau = 5.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_30days.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_30days.hdf"),
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
                ]),
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM_legacy

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:0',
    )
    
    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_duration_bias.hdf"    
    test_background_output = "testing_boutput_duration_bias.hdf"



### NEXT RUNS ###
# 1. Uniform on q (DEIMOS) - after Butterball finishes
# 2. Bias runs (each for Sage and 1D Resnet-50)
    # 2a. Spectral bias with different frequencies (sin) and const tau (WIAY)
    # 2b. Bias based on signal duration with const freq and different tau (WIAY)
# 3. Running BEST on different training seeds - 2 runs
# 4. Norland D3 run on XPHM
# 5. Without PE point estimate