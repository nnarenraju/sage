"""Train a run from its directory: record the settings, build every component, train.

The components are built in the old driver's order with its seeding, so a run
reproduces the old code bit for bit until per-stream seeds replace it.
"""

import dataclasses
import json
from pathlib import Path

import torch
import yaml

from sage.config import load_run, prepare_run_dir
from sage.config.machine import fiducial_dir, read_data_dir
from sage.config.prior import load_prior
from sage.config.run_dir import record_derived, record_fiducial, record_versions, training_started
from sage.core.logger import HDF5LossLogger, get_logger, setup_logging
from sage.core.pipeline import Grid, ProcessingState
from sage.data.asd import read_fiducial_asds
from sage.dsp.multirate_sampling import MultirateSampler
from sage.factory.build import (
    build_data,
    build_loss,
    build_model,
    build_noise,
    build_noise_stream,
    build_optimiser,
    build_preprocess,
    build_schedule,
    build_signal,
    check_heads,
    check_mining,
    check_train,
    check_validation,
    compile_network,
    flat_tail_start_epoch,
    mining_epochs,
    precision_setup,
    resolve_settings,
    validates,
)
from sage.factory.callbacks import EMACallback, HardMiningCallback
from sage.factory.ema_calibration import calibration_inputs, recalibrate_batchnorm
from sage.factory.training import SageVanillaTraining
from sage.factory.validation import SageVanillaValidation
from sage.utils.checkpoint import CheckpointManager, atomic_torch_save, next_epoch, rng_state, set_rng_state

logger = get_logger(__name__)

# A run resuming at epoch K seeds its samplers and the global generator with
# seed + SEED_STRIDE * K, so no stream replays its earlier draws (old driver).
SEED_STRIDE = 1_000_003
RECOLOUR_SEED_OFFSET = 7  # set in build_noise


def derived_values(settings: dict, window, processor, signal) -> dict:
    """
    Values computed from the settings, recorded in ``resolved/derived.yaml``.

    Parameters
    ----------
    settings : dict
        Settings from ``load_run``.
    window : Window
        Window geometry.
    processor : Preprocessor
        Preprocessing chain (its multirate bins).
    signal : torch.nn.Module
        Training signal sampler (its target scaling).

    Returns
    -------
    dict
        ``window``, ``multirate_bins``, ``target_scaling`` (offset and scale
        per target), ``flat_tail_start_epoch`` and ``seeds``.
    """
    targets = settings["signal"]["targets"]
    sampler = signal.param_sampler
    seeds = {
        "train": settings["train"]["seed"],
        "resume_stride": SEED_STRIDE,
        "recolour_offset": RECOLOUR_SEED_OFFSET,
        "validation": settings["validation"]["seed"],
    }
    if settings["mining"]["enabled"]:
        seeds["mining"] = settings["train"]["seed"]
    return {
        "window": dataclasses.asdict(window),
        "multirate_bins": [step.bins.tolist() for step in processor.seq if isinstance(step, MultirateSampler)],
        "target_scaling": {
            target: {"offset": float(sampler._std_means[i]), "scale": float(sampler._std_stds[i])}
            for i, target in enumerate(targets)
        },
        "flat_tail_start_epoch": flat_tail_start_epoch(settings["schedule"], settings["train"]["epochs"]),
        "seeds": seeds,
    }


def use_recorded_scaling(run_dir: Path, signal, targets: list) -> None:
    """
    Train a resumed segment against the target scaling recorded at the start.

    The z-score constants are Monte Carlo estimates from the training prior
    sampler, which a resumed segment reseeds, so recomputing them would move
    the targets' scaling at every resume (~0.1% at 10^6 draws).

    Parameters
    ----------
    run_dir : Path
        Run directory whose ``resolved/derived.yaml`` holds the scaling.
    signal : torch.nn.Module
        Training signal sampler.
    targets : list of str
        Targets, in head order.
    """
    scaling = yaml.safe_load((run_dir / "resolved" / "derived.yaml").read_text())["target_scaling"]
    sampler = signal.param_sampler
    for buffer, key in ((sampler._std_means, "offset"), (sampler._std_stds, "scale")):
        buffer.copy_(torch.tensor([scaling[target][key] for target in targets], dtype=buffer.dtype))


def train_run(run_dir: Path, accept_code_change: bool = False) -> None:
    """
    Train the run in ``run_dir`` from ``run.yaml``, resuming from its latest checkpoint.

    Parameters
    ----------
    run_dir : Path
        Run directory holding ``run.yaml``.
    accept_code_change : bool
        Continue a started run on changed Sage code.

    Raises
    ------
    ValueError
        On invalid settings, a heads / loss mismatch, settings or derived
        values changed after training started, or a code change without
        ``accept_code_change``.
    """
    run_dir = Path(run_dir)
    settings = load_run(run_dir / "run.yaml")
    train, validation, mining, ema = settings["train"], settings["validation"], settings["mining"], settings["ema"]
    check_train(train)
    check_validation(validation, train["batch_size"])
    check_heads(settings["model"], settings["loss"])
    if mining["enabled"]:
        check_mining(mining, settings["noise"], settings["model"], settings["schedule"], train["epochs"])
    prior = load_prior(settings["signal"]["prior"], run_dir)
    prepare_run_dir(run_dir, resolve_settings(settings), prior, resolve_settings)

    setup_logging(run_dir)
    start_epoch = next_epoch(run_dir)
    record_versions(run_dir, start_epoch, train["device"], accept_code_change)
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)

    window, detectors, dtype = build_data(settings["data"])
    device = torch.device(train["device"])
    targets = list(settings["signal"]["targets"])
    data_dir = read_data_dir()
    fiducial = record_fiducial(run_dir, fiducial_dir(data_dir, settings["noise"]["fiducial"]), detectors)
    asds = read_fiducial_asds(fiducial, detectors, window, device)

    batches = train["samples_per_epoch"] // train["batch_size"]
    signal_batch_size = round(train["batch_size"] * train["signal_fraction"])  # whole: check_train
    seed = train["seed"] + SEED_STRIDE * start_epoch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Run %s from epoch %d, seed %d", run_dir, start_epoch, seed)

    # The old driver's order: the global generator, seeded above, also draws the
    # model's initial weights, so everything before the model is built first.
    signal = build_signal(
        settings["signal"], prior, window, detectors, asds, signal_batch_size, seed, device, dtype
    )
    if training_started(run_dir):
        use_recorded_scaling(run_dir, signal, targets)
    noise, validation_noise = build_noise(
        settings["noise"], data_dir, window, detectors, train["batch_size"], seed, validation["seed"],
        mining["replay"]["start_jitter_in_s"], mining["replay"]["focal_gamma"], device, dtype,
    )
    validation_signal = build_signal(
        {**settings["signal"], "snr": validation["snr"]}, prior, window, detectors, asds,
        signal_batch_size, validation["seed"], device, dtype,
    )
    processor = build_preprocess(
        settings["preprocess"], asds, window, signal.param_sampler.bounds,
        settings["signal"]["low_frequency_cutoff_in_hz"], ProcessingState(Grid.FD_UNIFORM), device, dtype,
    )
    network = build_model(settings["model"], len(detectors), len(targets), device, dtype)
    network = network.to(memory_format=torch.channels_last)
    model = compile_network(network, train)
    loss = build_loss(settings["loss"], targets)
    optimiser = build_optimiser(settings["optimiser"], network)
    schedule, step_every = build_schedule(settings["schedule"], optimiser, train["epochs"], batches)
    autocast_dtype, scaler = precision_setup(train)
    record_derived(run_dir, derived_values(settings, window, processor, signal))

    callbacks = []
    if mining["enabled"]:
        # The training seed in every segment (old driver): the miner reseeds each round
        # from it and the epoch, so a resumed run mines as an uninterrupted one.
        miner = HardMiningCallback(
            mining, run_dir / "mining" / "bank.h5", mining_epochs(mining, settings["schedule"], train["epochs"]),
            train["seed"], detectors, window.sample_rate_in_hz,
        )
        callbacks.append(miner)
    averager = None
    if ema["enabled"]:
        averager = EMACallback(network, ema["decay"])
        callbacks.append(averager)
    trainer = SageVanillaTraining(
        signal, noise, processor, model, loss, optimiser, schedule, scaler,
        num_iterations=batches, num_epochs=train["epochs"], batch_size=train["batch_size"],
        signal_batch_size=signal_batch_size, num_estimates=len(targets), clip_norm=train["clip_norm"],
        device=device, dtype=dtype, autocast_dtype=autocast_dtype, scheduler_mode=step_every, callbacks=callbacks,
    )

    def validator(signal_sampler, noise_sampler, samples, num_epochs):
        return SageVanillaValidation(
            signal_sampler, noise_sampler, processor, model, loss,
            num_iterations=samples // train["batch_size"], num_epochs=num_epochs,
            batch_size=train["batch_size"], signal_batch_size=signal_batch_size,
            num_estimates=len(targets), device=device, dtype=dtype, autocast_dtype=autocast_dtype,
            results_path=run_dir / "results" / "validation.h5",
        )

    def fresh_samplers(role):
        """The training or validation samplers rebuilt from their seeds (no mined-window replay)."""
        streams = {
            "training": (train["seed"], settings["signal"]["snr"]),
            "validation": (validation["seed"], validation["snr"]),
        }
        stream_seed, snr = streams[role]
        fresh_signal = build_signal(
            {**settings["signal"], "snr": snr}, prior, window, detectors, asds, signal_batch_size, stream_seed,
            device, dtype,
        )
        fresh_noise = build_noise_stream(
            settings["noise"], role, data_dir, window, detectors, train["batch_size"], stream_seed,
            mining["replay"]["start_jitter_in_s"], mining["replay"]["focal_gamma"], device, dtype,
        )
        return fresh_signal, fresh_noise

    validate = validator(validation_signal, validation_noise, validation["samples"], train["epochs"])
    checkpoints = CheckpointManager(
        run_dir, network, optimiser, schedule, scaler, None if averager is None else averager.ema_model,
        train["keep_checkpoints"],
    )
    losses = HDF5LossLogger(run_dir / "results" / "losses.h5", train["epochs"], loss.component_names)
    if start_epoch > 0:
        start_epoch = checkpoints.load_latest(device)
        if mining["enabled"]:
            miner.attach_for_resume(trainer)

    for epoch in range(start_epoch, train["epochs"]):
        trainer(nepoch=epoch)
        losses.log(trainer.loss_components, epoch, split="training")
        validation_loss = None
        if validates(epoch, validation, train["epochs"]):
            validate(nepoch=epoch)
            losses.log(validate.loss_components, epoch, split="validation")
            validation_loss = float(validate.loss_components[epoch][0].item())
            trainer.metric_step(validation_loss)
        checkpoints.save(epoch, validation_loss)

    if next_epoch(run_dir) == train["epochs"] and not (run_dir / "results" / "final_check.json").exists():
        final_check(
            run_dir, settings, network, processor, loss, (noise, validation_noise), fresh_samplers, validator,
            signal_batch_size, device,
        )


def final_check(
    run_dir: Path, settings: dict, network, processor, loss, live_noise: tuple, fresh_samplers, validator,
    signal_batch_size: int, device,
) -> None:
    """
    Validate ``best.pt`` and, with EMA, the calibrated EMA weights on the same
    ``validation.final_samples`` set; write ``results/final_check.json``.

    The run's noise samplers are shut down first, freeing their memory. With
    EMA, the batch-norm statistics of ``ema.pt`` are recalibrated over
    ``ema.calibration_batches`` batches from fresh training samplers (the
    streams' first draws, no mined-window replay) and saved as
    ``ema_calibrated.pt``. Each checkpoint is validated on fresh validation
    samplers with the global generators restored, so both see the same batches.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    settings : dict
        Settings from ``load_run``.
    network : torch.nn.Module
        The uncompiled network; each checkpoint's weights are loaded into it.
    processor : Preprocessor
        Preprocessing chain.
    loss : torch.nn.Module
        The loss (its component names label the results).
    live_noise : tuple
        The run's training and validation noise samplers.
    fresh_samplers : callable
        ``role -> (signal, noise)``: the training or validation samplers
        rebuilt from their seeds.
    validator : callable
        ``(signal, noise, samples, num_epochs) -> SageVanillaValidation``.
    signal_batch_size : int
        Windows per batch that receive a signal.
    device : torch.device
        Device the checkpoints are loaded onto.
    """
    for sampler in live_noise:
        sampler.shutdown()
    checkpoints, ema = run_dir / "checkpoints", settings["ema"]
    result = {"samples": settings["validation"]["final_samples"]}
    groups = {"best.pt": "final_best"}
    if ema["enabled"]:
        state = torch.load(checkpoints / "ema.pt", map_location=device, weights_only=False)
        network.load_state_dict(state["model"])
        signal, noise = fresh_samplers("training")
        inputs = calibration_inputs(
            signal, noise, processor, signal_batch_size, len(settings["signal"]["targets"]),
            ema["calibration_batches"],
        )
        layers = recalibrate_batchnorm(network, inputs)
        noise.shutdown()
        atomic_torch_save({**state, "model": network.state_dict()}, checkpoints / "ema_calibrated.pt")
        groups["ema_calibrated.pt"] = "final_ema"

    rng = rng_state()
    for file, group in groups.items():
        state = torch.load(checkpoints / file, map_location=device, weights_only=False)
        network.load_state_dict(state["model"])
        set_rng_state(rng)
        signal, noise = fresh_samplers("validation")
        validate = validator(signal, noise, result["samples"], 1)
        validate(nepoch=0, name=group)
        noise.shutdown()
        losses = dict(zip(loss.component_names, validate.loss_components[0].tolist()))
        result[file] = {"epoch": state["epoch"], "loss": losses}

    if ema["enabled"]:
        result["ema_calibrated.pt"].update(
            batchnorm_layers_recalibrated=layers, calibration_batches=ema["calibration_batches"],
        )
        ema_total, best_total = (result[file]["loss"]["total"] for file in ("ema_calibrated.pt", "best.pt"))
        result["lower_total_loss"] = "ema_calibrated.pt" if ema_total <= best_total else "best.pt"
    (run_dir / "results" / "final_check.json").write_text(json.dumps(result, indent=1))
    logger.info(
        "Final check on %d validation samples: %s", result["samples"],
        "; ".join(f"{file} (epoch {result[file]['epoch']}) loss {result[file]['loss']['total']:.5f}" for file in groups),
    )
