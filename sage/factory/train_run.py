"""Train a run from its directory: record the settings, build every component, train.

Every random stream has its own seed (``sage.core.seed.stream_seed``).
"""

import dataclasses
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from sage.config import load_run, prepare_run_dir
from sage.config.machine import MissingData, dataset_file, fiducial_dir, read_data_dir
from sage.config.prior import load_prior
from sage.config.run_dir import (
    check_code,
    check_derived,
    check_record,
    record_derived,
    record_fiducial,
    record_versions,
    training_started,
)
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
    check_device,
    check_heads,
    check_mining,
    check_train,
    check_validation,
    compile_network,
    data_files,
    flat_tail_start_epoch,
    mining_epochs,
    precision_setup,
    resolve_settings,
    strain_runs,
    validates,
)
from sage.factory.callbacks import EMACallback, HardMiningCallback
from sage.factory.clone import check_built, check_start, clone_start, install_full_state, start_checkpoint
from sage.factory.ema_calibration import calibration_inputs, recalibrate_batchnorm
from sage.factory.training import SageVanillaTraining
from sage.factory.validation import SageVanillaValidation
from sage.core.seed import STREAMS, generator, stream_seed
from sage.utils.checkpoint import CheckpointManager, atomic_json_dump, atomic_torch_save, next_epoch, restore

logger = get_logger(__name__)

STOP_FILE = "STOP"  # written by `sage stop`; training stops before its next epoch
FINAL_CHECK_ESTIMATE = "final_check_estimate.json"  # in results/; written at every validation pass


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
        "validation": settings["validation"]["seed"],
        "streams": "numpy SeedSequence(seed, spawn_key=(stream, epoch of the segment; 0 for validation))",
        "stream_keys": dict(STREAMS),
    }
    return {
        "window": dataclasses.asdict(window),
        "multirate_bins": [step.bins.tolist() for step in processor.seq if isinstance(step, MultirateSampler)],
        "target_scaling": {
            target: {"offset": float(sampler._target_offsets[i]), "scale": float(sampler._target_scales[i])}
            for i, target in enumerate(targets)
        },
        "flat_tail_start_epoch": flat_tail_start_epoch(settings["schedule"], settings["train"]["epochs"]),
        "seeds": seeds,
    }


def recorded_scaling(run_dir: Path, start: dict | None) -> dict | None:
    """
    The target scaling a run trains against once it has one recorded.

    A started run keeps its own, and a clone the old run's, whatever the
    current code or prior would give.

    Parameters
    ----------
    run_dir : Path
        Run directory; a started run's ``resolved/derived.yaml`` holds it.
    start : dict or None
        A cloned run's start checkpoint, whose record holds it.

    Returns
    -------
    dict or None
        Offset and scale per target; None for a new run.
    """
    if training_started(run_dir):
        return yaml.safe_load((run_dir / "resolved" / "derived.yaml").read_text())["target_scaling"]
    return None if start is None else yaml.safe_load(start["record"]["derived"])["target_scaling"]


def use_scaling(signal, targets: list, scaling: dict) -> None:
    """Set the training signal sampler's target scaling (``recorded_scaling``), targets in head order."""
    sampler = signal.param_sampler
    for buffer, key in ((sampler._target_offsets, "offset"), (sampler._target_scales, "scale")):
        buffer.copy_(torch.tensor([scaling[target][key] for target in targets], dtype=buffer.dtype))


def check_run(run_dir: Path) -> tuple:
    """
    Load ``run.yaml`` and its prior and run every check on the settings.

    Parameters
    ----------
    run_dir : Path
        Run directory holding ``run.yaml``.

    Returns
    -------
    tuple
        ``(settings, prior)``, from ``load_run`` and ``load_prior``.

    Raises
    ------
    ValueError
        On invalid settings or a heads / loss mismatch.
    FileNotFoundError
        If the prior file is missing.
    """
    settings = load_run(Path(run_dir) / "run.yaml")
    train, mining = settings["train"], settings["mining"]
    check_train(train)
    check_validation(settings["validation"], train["batch_size"])
    check_heads(settings["model"], settings["loss"])
    if mining["enabled"]:
        check_mining(mining, settings["noise"], settings["model"], settings["schedule"], train["epochs"])
    return settings, load_prior(settings["signal"]["prior"], run_dir)


def check_data(run_dir: Path, settings: dict, data_dir: Path) -> None:
    """
    Raise if a file the run reads is missing, or its strain is at another sample rate.

    Parameters
    ----------
    run_dir : Path
        Run directory; a started run reads its own fiducial copy.
    settings : dict
        From ``check_run``.
    data_dir : Path
        The machine's data directory.

    Raises
    ------
    MissingData
        Naming every missing file.
    ValueError
        If an observing run's strain is at another sample rate than the run.
    """
    files = data_files(settings, data_dir, fiducial=not training_started(run_dir))
    missing = [path for path in files if not path.exists()]
    if missing:
        raise MissingData(f"{len(missing)} of the {len(files)} data files this run reads are missing:\n"
                          + "\n".join(f"  {path}" for path in missing))
    sample_rate = settings["data"]["sample_rate_in_hz"]
    for run in strain_runs(settings["noise"]):
        recorded = yaml.safe_load(dataset_file(data_dir, run).read_text())["sample_rate_in_hz"]
        if recorded != sample_rate:
            raise ValueError(f"{run} strain is at {recorded} Hz; data.sample_rate_in_hz is {sample_rate}")


def build_run(run_dir: Path, settings: dict, prior: dict, data_dir: Path, fiducial: Path, start_epoch: int,
              scaling: dict | None):
    """
    Build every component of a run; write nothing.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    settings, prior : dict
        From ``check_run``.
    data_dir : Path
        The machine's data directory.
    fiducial : Path
        Directory holding the fiducial ASDs.
    start_epoch : int
        Epoch the segment starts at; keys every training stream.
    scaling : dict or None
        Target scaling to train against (``recorded_scaling``); None keeps
        the training sampler's estimate.

    Returns
    -------
    SimpleNamespace
        The data geometry, samplers, preprocessing, network (``network``
        uncompiled, ``model`` as trained), loss, optimiser, schedule and
        precision, with the batch sizes and seed they were built with.
    """
    train, mining = settings["train"], settings["mining"]
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)

    window, detectors, dtype = build_data(settings["data"])
    device = torch.device(train["device"])
    targets = list(settings["signal"]["targets"])
    asds = read_fiducial_asds(fiducial, detectors, window, device)

    batches = train["samples_per_epoch"] // train["batch_size"]
    signal_batch_size = round(train["batch_size"] * train["signal_fraction"])  # whole: check_train
    seed = train["seed"]
    torch.manual_seed(stream_seed(seed, "global", start_epoch))  # also every GPU: model initialisation, dropout
    logger.info("Run %s from epoch %d, seed %d", run_dir, start_epoch, seed)

    signal = build_signal(
        settings["signal"], prior, window, detectors, asds, signal_batch_size, seed, start_epoch, device, dtype
    )
    if scaling is not None:
        use_scaling(signal, targets, scaling)
    noise = build_noise(
        settings["noise"], data_dir, window, detectors, train["batch_size"], seed, start_epoch,
        mining["replay"]["start_jitter_in_s"], mining["replay"]["focal_gamma"], device, dtype,
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
    return SimpleNamespace(
        window=window, detectors=detectors, dtype=dtype, device=device, targets=targets, asds=asds,
        batches=batches, signal_batch_size=signal_batch_size, seed=seed, signal=signal, noise=noise,
        processor=processor, network=network, model=model, loss=loss, optimiser=optimiser, schedule=schedule, step_every=step_every,
        autocast_dtype=autocast_dtype, scaler=scaler, placement=generator(seed, "placement", start_epoch, device),
    )


def fresh_streams(role: str, settings: dict, prior: dict, data_dir: Path, run) -> tuple:
    """
    The training or validation signal, noise and placement streams, rebuilt from their seed at epoch 0.

    Every validation pass, the final check and ``sage validate`` start from these, so they see the same
    windows (the final check's first ``validation.samples`` are a pass's); EMA calibration takes the
    training ones (no mined-window replay).

    Parameters
    ----------
    role : str
        ``training`` or ``validation``.
    settings, prior : dict
        From ``check_run``.
    data_dir : Path
        The machine's data directory.
    run : SimpleNamespace
        Window, detectors, ASDs, signal batch size, device and dtype (``build_run``, or the same fields).

    Returns
    -------
    tuple
        ``(signal, noise, placement)``.
    """
    train, replay = settings["train"], settings["mining"]["replay"]
    seed, snr = {
        "training": (train["seed"], settings["signal"]["snr"]),
        "validation": (settings["validation"]["seed"], settings["validation"]["snr"]),
    }[role]
    signal = build_signal(
        {**settings["signal"], "snr": snr}, prior, run.window, run.detectors, run.asds, run.signal_batch_size,
        seed, 0, run.device, run.dtype,
    )
    noise = build_noise_stream(
        settings["noise"], role, data_dir, run.window, run.detectors, train["batch_size"], seed, 0,
        replay["start_jitter_in_s"], replay["focal_gamma"], run.device, run.dtype,
    )
    return signal, noise, generator(seed, "placement", 0, run.device)


def final_check_seconds(settings: dict, pass_seconds: float, epoch_seconds: float, batches: int) -> float:
    """
    Time the final check takes, from one regular validation pass and one training epoch of the run.

    Each checkpoint's pass is ``final_samples / samples`` regular passes; with EMA a second pass, and the
    calibration's ``calibration_batches`` batches at the training rate.
    """
    validation, ema = settings["validation"], settings["ema"]
    seconds = (2 if ema["enabled"] else 1) * validation["final_samples"] / validation["samples"] * pass_seconds
    if ema["enabled"]:
        seconds += ema["calibration_batches"] / batches * epoch_seconds
    return seconds


def trainer_for(run, settings: dict, num_iterations: int, callbacks: list) -> SageVanillaTraining:
    """The trainer of a built run (``build_run``), ``num_iterations`` batches per epoch."""
    train = settings["train"]
    return SageVanillaTraining(
        run.signal, run.noise, run.processor, run.model, run.loss, run.optimiser, run.schedule, run.scaler,
        num_iterations=num_iterations, num_epochs=train["epochs"], batch_size=train["batch_size"],
        signal_batch_size=run.signal_batch_size, num_estimates=len(run.targets), placement=run.placement,
        clip_norm=train["clip_norm"],
        device=run.device, dtype=run.dtype, autocast_dtype=run.autocast_dtype, scheduler_mode=run.step_every,
        callbacks=callbacks,
    )


def train_run(run_dir: Path, accept_code_change: bool = False, stop_after: float | None = None) -> None:
    """
    Train the run in ``run_dir`` from ``run.yaml``, resuming from its latest checkpoint.

    Parameters
    ----------
    run_dir : Path
        Run directory holding ``run.yaml``.
    accept_code_change : bool
        Continue a started run on changed Sage code.
    stop_after : float or None
        Seconds after which to stop: no epoch starts once the time left is
        less than the longest epoch so far, and the final check does not start
        when it would not end in time (``final_check_seconds``). None trains to
        the end. Training
        also stops before an epoch while ``STOP_FILE`` exists in ``run_dir``.

    Raises
    ------
    ValueError
        On invalid settings or a heads / loss mismatch (``check_run``), or a
        device or precision not available here (``check_device``).
    MissingData
        If a file the run reads is missing from the data directory.
    SettingsChanged
        If the settings, prior or derived values changed after training started.
    CodeChanged
        On a code change without ``accept_code_change``.
    """
    clock = time.monotonic()
    run_dir = Path(run_dir)
    settings, prior = check_run(run_dir)
    train, validation, mining, ema = settings["train"], settings["validation"], settings["mining"], settings["ema"]
    check_device(train)
    data_dir = read_data_dir()
    check_data(run_dir, settings, data_dir)
    start, full_state = clone_start(run_dir) or (None, False)
    changes = None if start is None else check_start(resolve_settings(settings), start, full_state)
    prepare_run_dir(run_dir, resolve_settings(settings), prior, resolve_settings)

    setup_logging(run_dir)
    start_epoch = start["epoch"] + 1 if full_state else next_epoch(run_dir)
    if start is not None:
        logger.info("Starting from the cloned epoch %d (%s); settings changed: %s", start["epoch"],
                    "full state" if full_state else "weights only", "; ".join(changes) or "none")
    record_versions(run_dir, start_epoch, train["device"], accept_code_change)
    fiducial = record_fiducial(
        run_dir, fiducial_dir(data_dir, settings["noise"]["fiducial"]), settings["data"]["detectors"]
    )
    run = build_run(run_dir, settings, prior, data_dir, fiducial, start_epoch, recorded_scaling(run_dir, start))
    derived = derived_values(settings, run.window, run.processor, run.signal)
    if start is not None:
        check_built(derived, start, run.scaler, full_state)
    record_derived(run_dir, derived)
    if start is not None:
        # The EMA built below starts from these weights; full state then restores the rest as a resume.
        run.network.load_state_dict(start["model"])
        if full_state:
            install_full_state(run_dir, start, settings["ema"]["enabled"])

    callbacks = []
    if mining["enabled"]:
        # Not keyed by the segment: the miner reseeds each round from it and the
        # epoch, so a resumed run mines as an uninterrupted one.
        miner = HardMiningCallback(
            mining, run_dir / "mining" / "bank.h5", mining_epochs(mining, settings["schedule"], train["epochs"]),
            stream_seed(train["seed"], "mining"), run.detectors, run.window.sample_rate_in_hz,
        )
        callbacks.append(miner)
    averager = None
    if ema["enabled"]:
        averager = EMACallback(run.network, ema["decay"])
        callbacks.append(averager)
    trainer = trainer_for(run, settings, run.batches, callbacks)

    def validator(signal_sampler, noise_sampler, placement, samples, num_epochs):
        return SageVanillaValidation(
            signal_sampler, noise_sampler, run.processor, run.model, run.loss,
            num_iterations=samples // train["batch_size"], num_epochs=num_epochs,
            batch_size=train["batch_size"], signal_batch_size=run.signal_batch_size,
            num_estimates=len(run.targets), placement=placement, device=run.device, dtype=run.dtype,
            autocast_dtype=run.autocast_dtype, results_path=run_dir / "results" / "validation.h5",
        )

    checkpoints = CheckpointManager(
        run_dir, run.network, run.optimiser, run.schedule, run.scaler,
        None if averager is None else averager.ema_model, train["keep_checkpoints"],
    )
    losses = HDF5LossLogger(run_dir / "results" / "losses.h5", train["epochs"], run.loss.component_names)
    if start_epoch > 0:
        start_epoch = checkpoints.load_latest(run.device)
        if mining["enabled"]:
            miner.attach_for_resume(trainer)

    longest, estimate = 0.0, None
    for epoch in range(start_epoch, train["epochs"]):
        if (run_dir / STOP_FILE).exists():
            logger.info("Stopping before epoch %d: %s exists", epoch, run_dir / STOP_FILE)
            return
        elapsed = time.monotonic() - clock
        if stop_after is not None and elapsed + longest > stop_after:
            logger.info("Stopping before epoch %d: %.0f s left, the longest epoch took %.0f s",
                        epoch, stop_after - elapsed, longest)
            return
        begun = time.monotonic()
        trainer(nepoch=epoch)
        trained = time.monotonic() - begun
        losses.log(trainer.loss_components, epoch, split="training")
        validation_loss = None
        if validates(epoch, validation, train["epochs"]):
            # The same windows every pass: the validation streams start afresh.
            signal, noise, placement = fresh_streams("validation", settings, prior, data_dir, run)
            validate = validator(signal, noise, placement, validation["samples"], train["epochs"])
            passed = time.monotonic()
            try:
                validate(nepoch=epoch)
            finally:
                noise.shutdown()
            estimate = final_check_seconds(settings, time.monotonic() - passed, trained, run.batches)
            atomic_json_dump({"seconds": estimate, "epoch": epoch}, run_dir / "results" / FINAL_CHECK_ESTIMATE)
            losses.log(validate.loss_components, epoch, split="validation")
            validation_loss = float(validate.loss_components[epoch][0].item())
            trainer.metric_step(validation_loss)
        checkpoints.save(epoch, validation_loss)
        longest = max(longest, time.monotonic() - begun)

    if next_epoch(run_dir) == train["epochs"] and not (run_dir / "results" / "final_check.json").exists():
        left = None if stop_after is None else stop_after - (time.monotonic() - clock)
        if left is not None and estimate is not None and estimate > left:
            logger.info("Stopping before the final check: %.0f s left, it takes about %.0f s; the next job runs it",
                        left, estimate)
            return
        final_check(
            run_dir, settings, run.network, run.processor, run.loss, run.noise,
            lambda role: fresh_streams(role, settings, prior, data_dir, run), validator, run.signal_batch_size,
            run.device,
        )


def check_training(run_dir: Path, accept_code_change: bool = False, cpu: bool = False) -> float:
    """
    Run every start-up check of ``train_run``, then one real training step; write nothing.

    A started run is checked as its next segment would be (record, code,
    derived values) and the step starts from its latest checkpoint. The step
    trains without mining, which would write the bank.

    Parameters
    ----------
    run_dir : Path
        Run directory holding ``run.yaml``.
    accept_code_change : bool
        Accept a started run's code change, as ``train_run``.
    cpu : bool
        Take the step on the CPU and uncompiled, instead of ``train.device``.

    Returns
    -------
    float
        The step's total loss.

    Raises
    ------
    As ``train_run``.
    """
    run_dir = Path(run_dir)
    settings, prior = check_run(run_dir)
    started = training_started(run_dir)
    if started:
        check_record(run_dir, resolve_settings(settings), prior, resolve_settings)
        check_code(run_dir, accept_code_change)
    start, full_state = clone_start(run_dir) or (None, False)
    if start is not None:
        check_start(resolve_settings(settings), start, full_state)
    if cpu:
        settings = {**settings, "train": {**settings["train"], "device": "cpu", "compile": False}}
    check_device(settings["train"])
    data_dir = read_data_dir()
    check_data(run_dir, settings, data_dir)
    source = fiducial_dir(data_dir, settings["noise"]["fiducial"])
    fiducial = record_fiducial(run_dir, source, settings["data"]["detectors"]) if started else source
    start_epoch = start["epoch"] + 1 if full_state else next_epoch(run_dir)
    run = build_run(run_dir, settings, prior, data_dir, fiducial, start_epoch, recorded_scaling(run_dir, start))
    try:
        derived = derived_values(settings, run.window, run.processor, run.signal)
        if started:
            check_derived(run_dir, derived)
        if start is not None:
            check_built(derived, start, run.scaler, full_state)
            run.network.load_state_dict(start["model"])
        ema = settings["ema"]
        callbacks = [EMACallback(run.network, ema["decay"])] if ema["enabled"] else []
        ema_model = callbacks[0].ema_model if callbacks else None
        if started:
            CheckpointManager(
                run_dir, run.network, run.optimiser, run.schedule, run.scaler, ema_model,
                settings["train"]["keep_checkpoints"],
            ).load_latest(run.device)
        elif full_state:
            restore(start, run.network, run.optimiser, run.schedule, run.scaler, ema_model)
        trainer = trainer_for(run, settings, 1, callbacks)
        epoch = min(start_epoch, settings["train"]["epochs"] - 1)  # a finished run steps at its last epoch
        trainer(nepoch=epoch)
        fresh_streams("validation", settings, prior, data_dir, run)[1].shutdown()  # what every validation pass builds
    finally:
        run.noise.shutdown()
    return float(trainer.loss_components[epoch][0])


def final_check(
    run_dir: Path, settings: dict, network, processor, loss, live_noise, fresh_samplers, validator,
    signal_batch_size: int, device,
) -> None:
    """
    Validate ``best.pt`` and, with EMA, the calibrated EMA weights on the same
    ``validation.final_samples`` set; write ``results/final_check.json``.

    The run's noise sampler is shut down first, freeing its memory. With
    EMA, the batch-norm statistics of ``ema.pt`` are recalibrated over
    ``ema.calibration_batches`` batches from fresh training samplers (the
    streams' first draws, no mined-window replay) and saved as
    ``ema_calibrated.pt``. Each checkpoint is validated on fresh validation
    streams, so both see the same batches.

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
    live_noise : torch.nn.Module
        The run's training noise sampler.
    fresh_samplers : callable
        ``role -> (signal, noise, placement)`` (``fresh_streams``).
    validator : callable
        ``(signal, noise, placement, samples, num_epochs) -> SageVanillaValidation``.
    signal_batch_size : int
        Windows per batch that receive a signal.
    device : torch.device
        Device the checkpoints are loaded onto.
    """
    live_noise.shutdown()
    checkpoints, ema = run_dir / "checkpoints", settings["ema"]
    result = {"samples": settings["validation"]["final_samples"]}
    groups = {"best.pt": "final_best"}
    if ema["enabled"]:
        state = torch.load(checkpoints / "ema.pt", map_location=device, weights_only=False)
        network.load_state_dict(state["model"])
        signal, noise, placement = fresh_samplers("training")
        inputs = calibration_inputs(
            signal, noise, processor, signal_batch_size, len(settings["signal"]["targets"]),
            ema["calibration_batches"], placement,
        )
        layers = recalibrate_batchnorm(network, inputs)
        noise.shutdown()
        atomic_torch_save({**state, "model": network.state_dict()}, checkpoints / "ema_calibrated.pt")
        groups["ema_calibrated.pt"] = "final_ema"

    for file, group in groups.items():
        state = torch.load(checkpoints / file, map_location=device, weights_only=False)
        network.load_state_dict(state["model"])
        signal, noise, placement = fresh_samplers("validation")
        validate = validator(signal, noise, placement, result["samples"], 1)
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


def validate_checkpoint(run_dir: Path, from_epoch: int | None, ema_calibrated: bool,
                        accept_code_change: bool = False) -> dict:
    """
    Validate one stored checkpoint on ``validation.final_samples`` windows of the run's validation set.

    The validation streams are rebuilt from their seed, so a checkpoint always
    gives the same result, on the final check's windows. The pass is
    added to ``results/validate.h5``, which training never writes, so a running
    job is not disturbed.

    Parameters
    ----------
    run_dir : Path
        A started run directory.
    from_epoch : int or None
        Stored epoch (``epoch_NNNN.pt`` or ``latest.pt``); None takes ``best.pt``.
    ema_calibrated : bool
        Take ``ema_calibrated.pt`` (written by the final check) instead.
    accept_code_change : bool
        Validate on changed Sage code.

    Returns
    -------
    dict
        ``file``, ``epoch``, ``samples``, ``group`` and ``loss`` (component -> value).

    Raises
    ------
    FileNotFoundError
        If the run has no checkpoint, or not the one asked for.
    EpochUnavailable
        If ``from_epoch`` is not stored.
    SettingsChanged, CodeChanged
        As ``train_run``.
    """
    run_dir = Path(run_dir)
    settings, prior = check_run(run_dir)
    if not training_started(run_dir):
        raise FileNotFoundError(f"{run_dir}: no checkpoint to validate")
    check_record(run_dir, resolve_settings(settings), prior, resolve_settings)
    check_code(run_dir, accept_code_change)
    train, validation = settings["train"], settings["validation"]
    check_device(train)
    data_dir = read_data_dir()
    check_data(run_dir, settings, data_dir)
    if ema_calibrated:
        path, group = run_dir / "checkpoints" / "ema_calibrated.pt", "validate_ema_calibrated"
        if not path.exists():
            raise FileNotFoundError(f"{run_dir}: no ema_calibrated.pt; the final check writes it when EMA is enabled")
    elif from_epoch is None:
        path, group = run_dir / "checkpoints" / "best.pt", "validate_best"
    else:
        path, group = start_checkpoint(run_dir, from_epoch)[1], f"validate_epoch_{from_epoch:04d}"

    window, detectors, dtype = build_data(settings["data"])
    device = torch.device(train["device"])
    targets = list(settings["signal"]["targets"])
    fiducial = record_fiducial(run_dir, fiducial_dir(data_dir, settings["noise"]["fiducial"]), detectors)
    asds = read_fiducial_asds(fiducial, detectors, window, device)
    network = build_model(settings["model"], len(detectors), len(targets), device, dtype)
    network = network.to(memory_format=torch.channels_last)
    state = torch.load(path, map_location=device, weights_only=False)
    network.load_state_dict(state["model"])
    model = compile_network(network, train)
    loss = build_loss(settings["loss"], targets)
    signal_batch_size = round(train["batch_size"] * train["signal_fraction"])
    streams = SimpleNamespace(
        window=window, detectors=detectors, asds=asds, signal_batch_size=signal_batch_size, device=device, dtype=dtype,
    )
    signal, noise, placement = fresh_streams("validation", settings, prior, data_dir, streams)
    processor = build_preprocess(
        settings["preprocess"], asds, window, signal.param_sampler.bounds,
        settings["signal"]["low_frequency_cutoff_in_hz"], ProcessingState(Grid.FD_UNIFORM), device, dtype,
    )
    samples = validation["final_samples"]
    validate = SageVanillaValidation(
        signal, noise, processor, model, loss, num_iterations=samples // train["batch_size"], num_epochs=1,
        batch_size=train["batch_size"], signal_batch_size=signal_batch_size, num_estimates=len(targets),
        placement=placement, device=device, dtype=dtype,
        autocast_dtype=precision_setup(train)[0], results_path=run_dir / "results" / "validate.h5",
    )
    try:
        validate(nepoch=0, name=group)
    finally:
        noise.shutdown()
    losses = dict(zip(loss.component_names, validate.loss_components[0].tolist()))
    return {"file": path.name, "epoch": state["epoch"], "samples": samples, "group": group, "loss": losses}
