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
from architectures.models import Rigatoni_MS_ResNetCBAM
from architectures.models import KappaModel_ResNet_small
from architectures.frontend import MultiScaleBlock
# Transforms, augmentation and generation
from data.transforms import Unify, UnifySignal, UnifyNoise, UnifySignalGen, UnifyNoiseGen
from data.transforms import Whiten, MultirateSampling, Normalise
from data.transforms import AugmentOptimalNetworkSNR
from data.transforms import Recolour
from data.transforms import FastGenerateWaveform, RandomNoiseSlice, MultipleFileRandomNoiseSlice
# Loss functions
from losses.custom_loss_functions import BCEWithPEregLoss

# RayTune
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# TASKS
# Remove unwanted architectures from zoo
# All tmp files must be placed in a single location (eg. segments.csv)
# code to produce all tmp files must be consolidated
# code to download noise files for full experimentation must be consolidated
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
                                         fix_epoch = False,
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
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
class SageNetOTF_Feb24_Yukon(SageNetOTF):
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
class SageNetOTF_Feb24_Yukon_rerun(SageNetOTF):
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

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         fix_epoch = False,
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
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
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_30days.hdf"),
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


# RUNNING (I actually have high hopes for this one)
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
                                         fix_epoch = False,
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
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


# ABLATION 1 - small network resnet50
class SageNetOTF_metric_lowvar_Butterball(SageNetOTF):
    ### Primary Deviations (Comparison to BOY latest) ###
    # 1. 113 days of O3b data (not variation)
    # 2. SNR halfnorm (not variation)
    # 3. Template placement density (**VARIATION**)
    # 4. Low variation via fixed size dataset and lack of augmentation 

    """ Data storage """
    name = "SageNet50_metric_lowvar_Jul00"
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
                                         fix_epoch = False,
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
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
    model = KappaModel_ResNet_small

    model_params = dict(
        # Resnet50
        timm_params = {'model_name': 'resnet50',
                       'pretrained': False, 
                       'in_chans': 2, 
                       'drop_rate': 0.25
                      },
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'
    
    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_metric_small.hdf"    
    test_background_output = "testing_boutput_metric_small.hdf"


# RUNNING (High hopes as well)
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
                                         fix_epoch = False,
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
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


# RUNNING
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
                                         fix_epoch = False,
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
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


### POTENTIAL RUNS ###
# 1. uniform on (mchirp, q)
# 2. uniform on (tau0, q)
# 3. power law on (tau0, q)
# 4. uniform + power law on (tau0, q)
# 5. Anneal from U(m1, m2) to template placement metric
# 6. Anneal from template placement metric to U(m1, m2)