#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
o3b run script for the *combined* configuration: multi-detector consistency
model **plus** per-epoch hard-noise mining.

This is ``train_consistency.py`` (consistency model + merged-PE main loss +
per-detector ``ConsistencyNLLLoss`` aux under a ``GradientNormBalancer`` + the
``MaskingCallback`` 4-class non-astro assembly) with one addition: a
``HardMiningCallback`` that, after each epoch past its warmup, mines noise the
model scores as signal-like (using the consistency model's own attention
embedding as the QD diversity descriptor), accumulates the unique hard windows,
and replays them during training via the noise sampler.

Both add-ons are plain callbacks/aux-losses on :class:`SageVanillaTraining` —
there is no bespoke trainer; this script just wires them together.

Launch with ``python3 -c "from train_all import run_all_sage; run_all_sage()"``.
"""

import os
import torch

torch.backends.cudnn.benchmark = True

from sage.core.config import get_cfg, get_data_cfg

from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2, HalfNorm
from sage.data.waveform.snr import OptimalSNRRescaler
from sage.data.noise import MemmapNoiseSampler, RecolourPostprocess
from sage.data.non_astrophysical import NonAstrophysicalMasker

from sage.dsp.whiten import FiducialWhitening
from sage.dsp.multirate_sampling import MultirateSampler, DyadicPyramidBinning
from sage.core.graph import Preprocessor

from sage.architecture.network import MSCNN1D_2DResNetCBAM_Consistency
from sage.architecture.custom_losses import (
    BCEWithPEsigmaLoss, ConsistencyNLLLoss, GradientNormBalancer,
)
from sage.core.logger import HDF5LossLogger

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sage.factory import (
    SageVanillaTraining, MergedLossAdapter, ConsistencyLossAdapter, MaskingCallback,
)
from sage.factory.callbacks import HardMiningCallback

from config import set_configs
from sage.utils.checkpoint import CheckpointManager
from pathlib import Path

_RECOLOUR_DIR = "/local/scratch/igr/nnarenraju/data_release/o3a_dataset"
DATASET_DIR = Path("./hard_noise_datasets_all/")


def make_training_graph():
    # Signal sampler — append_per_det_targets=True so targets carry per-detector
    # tc and mchirp. `extra_batch` adds a pool of injections that the masker
    # turns into non-astrophysical class-0 pairs (dropped into noise slots), so
    # they never eat the coherent (class-1) signal budget.
    cfg = get_cfg()
    n_signal = int(cfg.batch_size * cfg.class_balance)
    n_noise = cfg.batch_size - n_signal
    extra_batch = min(round(cfg.p_non_astrophysical * n_noise), n_noise)

    param_sampler = read_from_config("./gwconfig.yaml", seed=150914)
    snrscaler = OptimalSNRRescaler(HalfNorm(scale=4.0, loc=5.0, seed=150914))
    signal_sampler = IMRPhenomPv2(
        param_sampler,
        ConstantProjection(),
        augment=snrscaler,
        append_per_det_targets=True,
        extra_batch=extra_batch,
    )

    # Noise sampler is built with the hard-replay hooks (training=True,
    # hard_dataset_dir, hard_bias_prob) so the HardMiningCallback can push mined
    # windows back via set_hard_dataset and they get replayed during training.
    recolour = RecolourPostprocess(p_recolour=0.37, recolour_dataset_dir=_RECOLOUR_DIR)
    noise_sampler = MemmapNoiseSampler(
        postprocess_fn   = recolour,
        prefetch         = 8,
        seed             = 150914,
        training         = True,
        hard_dataset_dir = str(DATASET_DIR),
        hard_bias_prob   = 0.6,
    )

    return signal_sampler, noise_sampler, param_sampler.bounds


def make_processor(bounds):
    # Build the multirate stage once; reuse its time grid for the heads so the
    # tc soft-argmax t_position matches the actual net_input.
    whitener = FiducialWhitening()
    dyadic_binning = DyadicPyramidBinning(bounds)
    mrsampler = MultirateSampler(binning_method=dyadic_binning)
    processor = Preprocessor([whitener, mrsampler])
    return processor, mrsampler.output_time_grid()


def run_all_sage():

    set_configs()
    cfg, data_cfg = get_cfg(), get_data_cfg()

    os.makedirs(cfg.export_dir, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    signal_sampler, noise_sampler, bounds = make_training_graph()
    processor, t_grid = make_processor(bounds)

    model = MSCNN1D_2DResNetCBAM_Consistency(
        t_grid,
        frontend_filters=32,
        frontend_kernel=64,
        backend_resnet_size=50,
        norm_type="groupnorm",
        dropout=cfg.dropout,
    ).to(dtype=cfg.dtype, device=cfg.device, memory_format=torch.channels_last)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    # max-autotune-no-cudagraphs (not plain max-autotune): the consistency model's
    # graph partitions on non-GPU / DeviceCopy ops, and capturing that partitioned
    # graph into cudagraph trees segfaults. Drop cudagraphs (keep Triton kernel
    # autotuning); the per-iter CPU-launch overhead is negligible at this size.
    model = torch.compile(
        model, mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=True
    )
    print("Model compiled with torch.compile!")

    merged_loss = BCEWithPEsigmaLoss(regression_weight=0.005, coupling_weight=0.005)
    consistency_loss = ConsistencyNLLLoss(tc_weight=1.0, mc_weight=1.0)

    optimiser = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6, fused=True)
    scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler(cfg.device, enabled=cfg.autocast)

    # Non-astrophysical (decoherent) sample generator — training only. Inert when
    # extra_batch == 0.
    masker = NonAstrophysicalMasker(
        delta_f=signal_sampler.df,
        tc_bounds=bounds["tc"],
        analysis_length_s=data_cfg.sample_length_in_s,
        seed=150914,
    )

    # All-of-it = vanilla training + per-detector aux loss + gradient-norm
    # balancer + non-astro masking callback + hard-noise mining callback.
    train_sage = SageVanillaTraining(
        signal_sampler,
        noise_sampler,
        processor,
        model,
        MergedLossAdapter(merged_loss),            # main: BCE + merged-PE
        optimiser,
        scheduler,
        scaler,
        num_iterations=cfg.training_iterations,
        num_epochs=cfg.num_epochs,
        aux_losses=[ConsistencyLossAdapter(consistency_loss)],   # per-det tc/mc NLL
        balancer=GradientNormBalancer(
            n_aux=4, balance_target=0.33, autocast=cfg.autocast,
            aux_names=["pe_reg", "coupling", "cons_tc", "cons_mc"],
        ),
        callbacks=[
            MaskingCallback(masker),               # 4-class non-astro assembly
            HardMiningCallback(                     # per-epoch hard-noise mining
                hard_bias_prob    = 0.6,
                # Keep noise scoring >= logit 2.0 (~88% signal probability).
                keep_threshold_raw = 2.0,
                warmup_epochs     = 10,
                mine_iters        = 200,
                hard_dataset_dir  = str(DATASET_DIR),
                max_total_samples = 30_000_000,
                mine_seed         = 150914,
            ),
        ],
    )

    ckpt_mgr = CheckpointManager(
        cfg=cfg, data_cfg=data_cfg, model=model,
        optimizer=optimiser, scheduler=scheduler, scaler=scaler,
    )

    logger = HDF5LossLogger(
        path=os.path.join(cfg.export_dir, "losses.h5"),
        num_epochs=cfg.num_epochs,
        num_components=train_sage.loss_components.shape[1],
    )

    for nepoch in range(cfg.num_epochs):
        print(f"Epoch {nepoch}: consistency + hard-mining training")
        train_sage(nepoch=nepoch)
        logger.log(train_sage.loss_components, nepoch, split="training")
        ckpt_mgr.save(
            epoch=nepoch,
            val_loss=float(train_sage.loss_components[nepoch][0].item()),
        )


if __name__ == "__main__":
    run_all_sage()
