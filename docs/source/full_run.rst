Production Run — Full Code
===========================

This page gives the complete, self-contained code needed to go from raw GWOSC data to a
trained Sage model. It mirrors the structure of ``runs/o3b/`` in the repository.

A production run has three distinct phases:

1. **Dataset preparation** — download and index noise data, estimate PSDs.
2. **Configuration** — define all hyperparameters in a config class.
3. **Training loop** — assemble all components and run epoch-by-epoch.

Phases 1 and 2 are one-time setup; phase 3 is the iterative training.

----

Phase 1: Dataset preparation
------------------------------

Save this as ``dataset.py`` (see ``runs/o3b/dataset.py``):

.. code-block:: python

    import math
    from sage.data.primer import DataReleaseDownloader, TimelineQuery, EstimatePSD, NoBlackout
    from sage.dsp.welch import TorchWelch
    from sage.data.noise import MemmapSingleNoiseSampler
    from sage.data.psd import smoothing
    from sage.core.config import get_cfg, get_data_cfg


    def _get_buffer(data_cfg):
        return math.ceil(0.2 * data_cfg.sample_rate) / data_cfg.sample_rate


    def _get_timeline(data_cfg):
        tq = TimelineQuery(
            detector=["H1", "L1"],
            observing_run=["O3b"],
            auto_clean_empty_timelines=True,
        )
        tq.download_segments()
        tq.prune_segments(
            rm_short_segments=True, rm_min_duration=22.0,
            rm_allevents=True, rm_window_length=30,
        )
        buffer = _get_buffer(data_cfg)
        tq.split_into_mini_segments(
            mini_segment_length=512.0 + buffer * 2.0,
            minimum_segment_duration=16.0,
        )
        tq.sanity_check_mini_segments(
            mini_segment_length=512.0 + buffer * 2.0,
            minimum_segment_duration=16.0,
            verbose=True,
        )
        return tq


    def _download(tq, data_cfg):
        buffer = _get_buffer(data_cfg)
        drd = DataReleaseDownloader(
            segments_metadata=tq.timeline,
            save_parent_dir="/work/sage/",
            noise_low_freq_cutoff=15.0,
            minimum_segment_duration=22.0,
            corrupt_trim_length=buffer,
            max_download_retries=15,
            retry_delay=5.0,
            num_workers=16,
            make_monolithic_file=True,
            sample_rate=data_cfg.sample_rate,
            save_bin=True,
        )
        drd.download()


    def _make_psds(detector, data_cfg):
        epsd = EstimatePSD(
            detector=detector,
            apply_inverse_spectrum_truncation=False,
            max_filter_len=int(round(2048.0 * 2)),
            low_frequency_cutoff=15.0,
            trunc_method="hann",
            psd_method=TorchWelch(
                delta_t=1 / 2048,
                seg_len=int(2048.0 * 4),
                seg_stride=int(2048.0 * 2),
                avg_method="median",
            ),
            num_samples=250_000,
            store_psds_as_bin=True,
            blackout_policy=NoBlackout(),
            interpolate_psd=True,
            training_sample_length=int(2048.0 * 16),
            psd_smoothener=smoothing.LogSplineSmoothing(
                smooth_factor=None,
                noise_low_frequency_cutoff=data_cfg.noise_low_frequency_cutoff,
            ),
        )
        bin_path = f"/work/sage/data_release/data_{detector}_O3b.bin"
        epsd.estimate_segment_psds(noise_segments_file=bin_path)
        epsd.estimate_raw_psds(
            noise_sampler=MemmapSingleNoiseSampler(bin_path, return_tensor=True),
            duration=int(round(2048.0 * 16)),
        )


    def make_dataset():
        cfg, data_cfg = get_cfg(), get_data_cfg()
        tq = _get_timeline(data_cfg)
        _download(tq, data_cfg)
        for det in ["H1", "L1"]:
            _make_psds(det, data_cfg)

----

Phase 2: Configuration
-----------------------

Save this as ``config.py`` (see ``runs/o3b/config.py``):

.. code-block:: python

    import torch
    from sage.core.config import register_configs
    from sage.core.base_classes import BaseConfig, BaseDataConfig


    class RunCFG:
        export_dir             = "./run_export"
        batch_size             = 128
        device                 = "cuda:0"
        dtype                  = torch.float32
        detectors              = ["H1", "L1"]
        do_point_estimate      = ["tc", "mchirp"]    # parameters to regress
        autocast               = True                 # mixed precision (float16)
        class_balance          = 0.5                  # fraction of batch = signals
        clip_norm              = 1.0
        num_epochs             = 80
        training_iterations    = int(2_000_000 / batch_size)    # ~15 625 batches/epoch
        validation_iterations  = int(200_000 / batch_size)      # ~1 562 batches/epoch


    class RunDataCFG:
        data_dir               = "/work/sage/data_dir"
        training_noise_files   = [
            "/work/sage/data_release/data_H1_O3b.bin",
            "/work/sage/data_release/data_L1_O3b.bin",
        ]
        validation_noise_files = [
            "/work/sage/o3a/data_release/data_H1_O3a.bin",
            "/work/sage/o3a/data_release/data_L1_O3a.bin",
        ]
        sample_rate                = 2048.0   # Hz
        noise_low_frequency_cutoff = 15.0     # Hz
        signal_low_frequency_cutoff= 20.0     # Hz
        sample_length_in_s         = 12.0     # seconds
        padding_length_in_s        = 2.0      # seconds


    def set_configs():
        register_configs(BaseConfig(RunCFG()), BaseDataConfig(RunDataCFG()))

.. note::

   ``training_noise_files`` points at O3b data; ``validation_noise_files`` points at
   O3a data.  Validating on a different run's noise is intentional — it tests
   out-of-distribution robustness without contaminating the training set.

----

Phase 3: Training loop
-----------------------

Save this as ``train.py`` (see ``runs/o3b/train.py``):

.. code-block:: python

    import os
    import torch

    torch._dynamo.config.verbose   = False
    torch._inductor.config.debug   = False
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    from sage.core.config import get_cfg, get_data_cfg

    from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2
    from sage.data.waveform import HalfNorm
    from sage.data.waveform.snr import OptimalSNRRescaler
    from sage.data.noise import MemmapNoiseSampler, RecolourPostprocess

    from sage.dsp.whiten import FiducialWhitening
    from sage.dsp.multirate_sampling import MultirateSampler, DyadicPyramidBinning
    from sage.core.graph import Preprocessor

    from sage.architecture.network import MSCNN1D_2DResNetCBAM_Heteroscedastic
    from sage.architecture.custom_losses import BCEWithPEsigmaLoss
    from sage.core.logger import HDF5LossLogger
    from sage.utils.checkpoint import CheckpointManager

    from sage.factory.training import SageUncompiledTraining
    from sage.factory.validation import SageUncompiledValidation

    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    from config import set_configs


    def make_training_graph():
        param_sampler   = read_from_config("./gwconfig.yaml", seed=150914)
        snrscaler       = OptimalSNRRescaler(HalfNorm(scale=4.0, loc=5.0, seed=150914))
        signal_sampler  = IMRPhenomPv2(param_sampler, ConstantProjection(), augment=snrscaler)
        noise_sampler   = MemmapNoiseSampler(
            postprocess_fn=RecolourPostprocess(p_recolour=0.37,
                                               recolour_dataset_dir="/work/sage/data_dir"),
            prefetch=8, seed=150914,
        )
        return signal_sampler, noise_sampler, param_sampler.bounds


    def make_validation_graph():
        param_sampler   = read_from_config("./gwconfig.yaml", seed=170817)
        snrscaler       = OptimalSNRRescaler(HalfNorm(scale=4.0, loc=5.0, seed=170817))
        signal_sampler  = IMRPhenomPv2(param_sampler, ConstantProjection(), augment=snrscaler)
        noise_sampler   = MemmapNoiseSampler(postprocess_fn=None, prefetch=8, seed=170817)
        return signal_sampler, noise_sampler


    def make_processor(bounds):
        return Preprocessor([
            FiducialWhitening(),
            MultirateSampler(binning_method=DyadicPyramidBinning(bounds)),
        ])


    def run_sage():

        set_configs()
        cfg, data_cfg = get_cfg(), get_data_cfg()

        train_sig, train_noise, bounds = make_training_graph()
        val_sig, val_noise             = make_validation_graph()
        processor                      = make_processor(bounds)

        # ── Model ──────────────────────────────────────────────────────────────
        model = MSCNN1D_2DResNetCBAM_Heteroscedastic(
            frontend_filters=32,
            frontend_kernel=64,
            backend_resnet_size=50,
            norm_type="instancenorm",
        ).to(dtype=cfg.dtype, device=cfg.device, memory_format=torch.channels_last)

        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

        model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)

        # ── Loss, optimiser, scheduler ─────────────────────────────────────────
        loss_fn   = BCEWithPEsigmaLoss(regression_weight=0.005, coupling_weight=0.005)
        optimiser = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6, fused=True)
        scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=5, T_mult=2, eta_min=1e-6)
        scaler    = torch.amp.GradScaler(cfg.device, enabled=cfg.autocast)

        # ── Training and validation loops ──────────────────────────────────────
        train_sage = SageUncompiledTraining(
            train_sig, train_noise, processor, model, loss_fn,
            optimiser, scheduler, scaler,
            num_iterations=cfg.training_iterations,
            num_epochs=cfg.num_epochs,
        )

        validate_sage = SageUncompiledValidation(
            val_sig, val_noise, processor, model, loss_fn,
            num_iterations=cfg.validation_iterations,
            num_epochs=cfg.num_epochs,
        )

        # ── Checkpointing and logging ──────────────────────────────────────────
        ckpt_mgr = CheckpointManager(
            cfg=cfg, data_cfg=data_cfg,
            model=model, optimizer=optimiser,
            scheduler=scheduler, scaler=scaler,
        )

        logger = HDF5LossLogger(
            path=os.path.join(cfg.export_dir, "losses.h5"),
            num_epochs=cfg.num_epochs,
            num_components=loss_fn.num_components,
        )

        # ── Main loop ──────────────────────────────────────────────────────────
        for nepoch in range(cfg.num_epochs):

            print(f"\nEpoch {nepoch}: Training")
            train_sage(nepoch=nepoch)
            logger.log(train_sage.loss_components, nepoch, split="training")

            if (nepoch + 1) % 5 == 0 or nepoch == 0:
                print(f"Epoch {nepoch}: Validating")
                validate_sage(nepoch=nepoch)
                logger.log(validate_sage.loss_components, nepoch, split="validation")

                val_loss = validate_sage.loss_components[nepoch][0].item()
                ckpt_mgr.save(epoch=nepoch, val_loss=val_loss)

----

Running the full pipeline
--------------------------

.. code-block:: bash

    # Step 1 — one-time dataset preparation (hours on a cluster node)
    python3 -c "from config import set_configs; set_configs(); \
                from dataset import make_dataset; make_dataset()"

    # Step 2 — training (80 epochs, ~2 M samples/epoch with batch 128)
    python3 -c "from train import run_sage; run_sage()"

Or use the provided shell script in ``runs/o3b/start.sh``:

.. code-block:: bash

    bash runs/o3b/start.sh

Export directory layout
------------------------

After a complete run, ``run_export/`` contains:

.. code-block:: text

    run_export/
    ├── checkpoints/
    │   ├── best.pt           ← lowest-val-loss checkpoint
    │   └── latest.pt         ← most recent checkpoint
    ├── losses.h5             ← per-epoch loss components (training + validation)
    ├── validation_data.h5    ← per-epoch network outputs and targets for diagnostics
    └── fiducial_psds/
        ├── fiducial_H1_psd.bin
        └── fiducial_L1_psd.bin
