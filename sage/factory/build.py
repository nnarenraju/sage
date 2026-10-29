"""Build run components from checked settings, one section at a time."""

import importlib.util
import inspect
import json
from pathlib import Path

import torch
from torch.optim import lr_scheduler

from sage.architecture.custom_losses import BCE_PE_Huber, BCE_PE_MSE, BCE_PE_Sigma
from sage.architecture.network import (
    MSCNN1D_2DResNetCBAM,
    MSCNN1Datt_2DResNetCBAM,
    MSCNN1Datt_catt_2DResNetCBAM,
)
from sage.architecture.network.common import FrontendTap
from sage.config.loader import component_arguments, convert, suggestion
from sage.config.machine import (
    asds_file,
    bad_time_file,
    dataset_file,
    fiducial_dir,
    fiducial_file,
    observing_run_dir,
    recolour_bank_file,
    segment_asd_file,
    segments_file,
    strain_file,
)
from sage.core.checks import (
    check_on_grid,
    check_tc_window,
    check_unique_detectors,
    check_whole_batches,
    check_whole_signals,
)
from sage.core.geometry import Window, window_from_data
from sage.core.graph import Preprocessor
from sage.core.logger import get_logger
from sage.core.pipeline import ProcessingState
from sage.core.seed import generator, stream_seed
from sage.data.noise.data_quality import noise_bad_segments
from sage.data.noise.real_noise import MemmapNoiseSampler
from sage.data.noise.recolour import RecolourPostprocess
from sage.data.noise.selection import check_disjoint, select_records
from sage.data.noise.white_noise import WhiteGaussianNoiseSampler, check_detector_asds
from sage.data.waveform import ConstantProjection, prior_sampler
from sage.data.waveform.approximants.IMRPhenomPv2 import IMRPhenomPv2
from sage.data.waveform.approximants.IMRPhenomXAS_NRTidalv3 import IMRPhenomXAS_NRTidalv3
from sage.data.waveform.approximants.IMRPhenomXPHM import IMRPhenomXPHM
from sage.data.waveform.distributions.snr_rescaling import HalfNorm
from sage.data.waveform.snr import OptimalSNRRescaler
from sage.dsp.multirate_sampling import DyadicPyramidBinning, MultirateSampler
from sage.dsp.normalise import MinMaxNormalise
from sage.dsp.whiten import FiducialWhitening
from sage.factory.schedulers import WarmupPolynomialDecayFlatTail

logger = get_logger(__name__)

DTYPES = {"float32": torch.float32, "float64": torch.float64}
WAVEFORMS = {
    "IMRPhenomPv2": IMRPhenomPv2,
    "IMRPhenomXPHM": IMRPhenomXPHM,
    "IMRPhenomXAS_NRTidalv3": IMRPhenomXAS_NRTidalv3,
}
PROJECTIONS = {"constant": ConstantProjection}
SNR_DISTRIBUTIONS = {"halfnorm": HalfNorm}
SNR_PROVIDED = {"seed", "device"}
PROJECTION_PROVIDED = {"detectors", "window", "device", "dtype", "generator"}
WAVEFORM_PROVIDED = {
    "param_sampler",
    "waveform_project",
    "augment",
    "window",
    "signal_batch_size",
    "low_frequency_cutoff_in_hz",
    "reference_frequency_in_hz",
    "targets",
    "device",
    "dtype",
    "append_per_det_targets",
}
PREPROCESS_STEPS = {
    "whiten": FiducialWhitening,
    "minmax_normalise": MinMaxNormalise,
    "multirate": MultirateSampler,
}
MODELS = {
    "MSCNN1D_2DResNetCBAM": MSCNN1D_2DResNetCBAM,
    "MSCNN1Datt_2DResNetCBAM": MSCNN1Datt_2DResNetCBAM,
    "MSCNN1Datt_catt_2DResNetCBAM": MSCNN1Datt_catt_2DResNetCBAM,
}
MODEL_PROVIDED = {"num_detectors", "num_estimates"}
LOSSES = {"BCE_PE_Sigma": BCE_PE_Sigma, "BCE_PE_Huber": BCE_PE_Huber, "BCE_PE_MSE": BCE_PE_MSE}
LOSS_PROVIDED = {"targets"}
OPTIMISERS = {
    name: cls
    for name, cls in vars(torch.optim).items()
    if inspect.isclass(cls) and issubclass(cls, torch.optim.Optimizer) and cls is not torch.optim.Optimizer
}
OPTIMISER_PROVIDED = {"params", "fused"}
OPTIMISER_DEFAULTS = {"AdamW": {"lr": 2e-4, "weight_decay": 1e-4}}  # production
SCHEDULES = {
    "WarmupPolynomialDecayFlatTail": WarmupPolynomialDecayFlatTail,
    **{
        name: cls
        for name, cls in vars(lr_scheduler).items()
        if inspect.isclass(cls)
        and issubclass(cls, lr_scheduler.LRScheduler)
        and cls is not lr_scheduler.LRScheduler
        and not name.startswith("_")
    },
}
SAGE_SCHEDULE_PROVIDED = {"optimizer", "total_steps", "steps_per_epoch"}
TORCH_SCHEDULE_PROVIDED = {"optimizer", "last_epoch", "verbose"}
STEP_EVERY = ("batch", "epoch", "metric")
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


def build_data(data: dict) -> tuple:
    """
    Derive what every component takes from the ``data`` settings.

    Parameters
    ----------
    data : dict
        Checked ``data`` section of a run file.

    Returns
    -------
    window : Window
        Window geometry.
    detectors : list of str
        Detectors, in channel order.
    dtype : torch.dtype
        Dtype of the data tensors.

    Raises
    ------
    ValueError
        If a detector is listed twice or a length is not whole samples.
    """
    check_unique_detectors(data["detectors"])
    window: Window = window_from_data(data)

    if data["dtype"] == "float64":
        logger.warning(
            "dtype float64: waveform models compute in float64 regardless; "
            "only their output and the data tensors follow dtype"
        )
    return window, list(data["detectors"]), DTYPES[data["dtype"]]


def lookup(table: dict, component: dict, where: str) -> tuple:
    """
    Find a component's class by name and split off its parameters.

    Parameters
    ----------
    table : dict
        Component names to classes.
    component : dict
        Component settings, ``name`` and parameters.
    where : str
        Dotted location, for error messages.

    Returns
    -------
    cls : type
        Component class.
    parameters : dict
        Settings other than ``name``.

    Raises
    ------
    ValueError
        If the name is unknown.
    """
    name = component["name"]
    if name not in table:
        raise ValueError(f"{where}.name: unknown {name!r}; {suggestion(name, list(table))}")
    return table[name], {key: value for key, value in component.items() if key != "name"}


def build_signal(
    signal: dict,
    prior: dict,
    window: Window,
    detectors: list,
    asds: torch.Tensor,
    signal_batch_size: int,
    seed: int,
    epoch: int,
    device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """
    Build the signal sampler: prior, waveform, projection and SNR.

    Parameters
    ----------
    signal : dict
        Checked ``signal`` section of a run file.
    prior : dict
        Checked prior.
    window : Window
        Window geometry.
    detectors : list of str
        Detectors, in channel order.
    asds : torch.Tensor, shape ``(D, F)``
        Fiducial ASDs, for the SNR.
    signal_batch_size : int
        Signals drawn per call.
    seed, epoch : int
        The prior, SNR and GMST streams are ``stream_seed(seed, <stream>, epoch)``.
    device : str or torch.device
        Device of the batches.
    dtype : torch.dtype
        Dtype of the data tensors.

    Returns
    -------
    torch.nn.Module
        Waveform model; each call returns strain and targets.

    Raises
    ------
    ValueError
        If a frequency is off the grid, the tc prior leaves the window, a
        target is not a prior parameter, or a component setting is wrong.
    """
    f_low = signal["low_frequency_cutoff_in_hz"]
    f_ref = signal["reference_frequency_in_hz"]
    check_on_grid("signal.low_frequency_cutoff_in_hz", f_low, window.padded_length_in_s)
    check_on_grid("signal.reference_frequency_in_hz", f_ref, window.padded_length_in_s)

    sampler = prior_sampler(prior, stream_seed(seed, "prior", epoch), device, dtype)
    check_tc_window(*sampler.bounds["tc"], window.sample_length_in_s)
    for target in signal["targets"]:
        if target not in sampler.param_index:
            raise ValueError(
                f"signal.targets: {target!r} is not a prior parameter; "
                f"{suggestion(target, list(sampler.param_index))}"
            )

    snr_cls, snr_parameters = lookup(SNR_DISTRIBUTIONS, signal["snr"], "signal.snr")
    distribution = snr_cls(
        **component_arguments(snr_cls, snr_parameters, SNR_PROVIDED, "signal.snr"),
        seed=stream_seed(seed, "snr", epoch),
        device=device,
    )
    rescaler = OptimalSNRRescaler(distribution, asds, window, f_low, device)

    projection_cls, projection_parameters = lookup(
        PROJECTIONS, signal["projection"], "signal.projection"
    )
    projection = projection_cls(
        **component_arguments(
            projection_cls, projection_parameters, PROJECTION_PROVIDED, "signal.projection"
        ),
        detectors=detectors,
        window=window,
        device=device,
        dtype=dtype,
        generator=generator(seed, "gmst", epoch, device),
    )

    waveform_cls, waveform_parameters = lookup(WAVEFORMS, signal["waveform"], "signal.waveform")
    return waveform_cls(
        **component_arguments(waveform_cls, waveform_parameters, WAVEFORM_PROVIDED, "signal.waveform"),
        param_sampler=sampler,
        waveform_project=projection,
        augment=rescaler,
        window=window,
        signal_batch_size=signal_batch_size,
        low_frequency_cutoff_in_hz=f_low,
        reference_frequency_in_hz=f_ref,
        targets=list(signal["targets"]),
        device=device,
        dtype=dtype,
        append_per_det_targets=False,
    )


def noise_runs(source: dict, data_dir: Path, detectors: list) -> list:
    """
    Strain files and ASD directory of every run a real noise source pools.

    Parameters
    ----------
    source : dict
        Checked ``noise.training`` or ``noise.validation`` source.
    data_dir : Path
        Directory from the machine file.
    detectors : list of str
        Detectors, in channel order.

    Returns
    -------
    list of dict
        ``{"run", "data_dir", "bins"}`` per run, for ``MemmapNoiseSampler``,
        with the source's ``portion`` or ``gps_range`` if it has one.
    """
    selection = {key: source[key] for key in ("portion", "gps_range") if key in source}
    return [
        {
            "run": run,
            "data_dir": observing_run_dir(data_dir, run, "asds"),
            "bins": [str(strain_file(data_dir, run, detector)) for detector in detectors],
            **selection,
        }
        for run in source["runs"]
    ]


def selected_records(run: dict, detectors: list) -> dict:
    """
    Segment records a noise run draws from, per detector.

    Parameters
    ----------
    run : dict
        One run from :func:`noise_runs`.
    detectors : list of str
        Detectors, in channel order.

    Returns
    -------
    dict of str to list of dict
        Selected sidecar records.
    """
    records = {}
    for detector, path in zip(detectors, run["bins"]):
        sidecar = segments_file(path)
        records[detector] = json.loads(sidecar.read_text())
    return select_records(records, run)


def build_recolour(
    recolour: dict, runs: list, data_dir: Path, window: Window, detectors: list,
    seed: int, device,
) -> RecolourPostprocess:
    """
    Build the recolour step for noise drawn from ``runs``.

    Parameters
    ----------
    recolour : dict
        Checked ``noise.recolour`` group.
    runs : list of dict
        Runs of the sampler it recolours, from :func:`noise_runs`.
    data_dir : Path
        Directory from the machine file.
    window : Window
        Window geometry.
    detectors : list of str
        Detectors, in channel order.
    seed : int
        Seed of the recolour draws.
    device : str or torch.device
        Device of the output.

    Returns
    -------
    RecolourPostprocess
        Recolour step.
    """
    return RecolourPostprocess(
        runs=runs,
        target_asd_dir=observing_run_dir(data_dir, recolour["target"], "asds"),
        probability=recolour["probability"],
        asd_jitter_gain=recolour["asd_jitter_gain"],
        seed=seed,
        detectors=detectors,
        window=window,
        device=device,
    )


def recoloured(noise: dict, role: str) -> bool:
    """
    Whether the ``role`` noise (``training`` or ``validation``) is recoloured.

    Raises
    ------
    ValueError
        If recolouring is asked of synthetic noise.
    """
    recolour = noise.get("recolour")
    on = recolour is not None and (role == "training" or recolour["validation"])
    if on and noise[role]["name"] != "real":
        raise ValueError(f"noise.recolour applies to real noise; noise.{role} is synthetic")
    return on


def strain_runs(noise: dict) -> list:
    """Observing runs whose strain the real noise sources read, sorted."""
    return sorted({run for role in ("training", "validation") if noise[role]["name"] == "real" for run in noise[role]["runs"]})


def asd_runs(noise: dict) -> list:
    """Observing runs whose per-segment ASDs the recoloured noise sources read, sorted."""
    return sorted({run for role in ("training", "validation") if recoloured(noise, role) for run in noise[role]["runs"]})


def bank_runs(noise: dict) -> list:
    """Observing runs whose recolour bank the run reads: the recolour target, if any noise is recoloured."""
    return [noise["recolour"]["target"]] if recoloured(noise, "training") or recoloured(noise, "validation") else []


def data_files(settings: dict, data_dir: Path, fiducial: bool = True) -> list:
    """
    Every file a run reads from the data directory, in a fixed order, without repeats.

    Parameters
    ----------
    settings : dict
        Checked run settings.
    data_dir : Path
        Directory from the machine file.
    fiducial : bool
        Include the fiducial ASDs; a started run reads its own copy.

    Returns
    -------
    list of Path
        Strain records, strain files with their sidecars and bad time, the recoloured runs' ASD records
        and per-segment ASDs and the recolour target's bank, then the fiducial ASDs.
    """
    noise, detectors = settings["noise"], settings["data"]["detectors"]
    files = [dataset_file(data_dir, run) for run in strain_runs(noise)]
    for role in ("training", "validation"):
        if noise[role]["name"] != "real":
            continue
        runs = noise_runs(noise[role], data_dir, detectors)
        for run in runs:
            for path in map(Path, run["bins"]):
                files += [path, segments_file(path), bad_time_file(path)]
        if recoloured(noise, role):
            for run in runs:
                files.append(asds_file(data_dir, run["run"]))
                for detector in detectors:
                    asds = segment_asd_file(run["data_dir"], detector)
                    files += [asds, segments_file(asds)]
            files.append(asds_file(data_dir, noise["recolour"]["target"]))
            target = observing_run_dir(data_dir, noise["recolour"]["target"], "asds")
            for detector in detectors:
                bank = recolour_bank_file(target, detector)
                files += [bank, bank.with_suffix(".json")]
    if fiducial:
        for detector in detectors:
            asd = fiducial_file(fiducial_dir(data_dir, noise["fiducial"]), detector)
            files += [asd, asd.with_suffix(".json")]
    return list(dict.fromkeys(files))


def build_noise_stream(
    noise: dict,
    role: str,
    data_dir: Path,
    window: Window,
    detectors: list,
    batch_size: int,
    seed: int,
    epoch: int,
    hard_start_jitter_s: float,
    hard_focal_gamma: float,
    device,
    dtype: torch.dtype,
):
    """
    Build the training or validation noise sampler alone.

    Parameters
    ----------
    noise : dict
        Checked ``noise`` section of a run file.
    role : str
        ``training`` or ``validation``.
    data_dir : Path
        Directory from the machine file.
    window : Window
        Window geometry.
    detectors : list of str
        Detectors, in channel order.
    batch_size : int
        Windows per batch.
    seed, epoch : int
        The noise and recolour streams are ``stream_seed(seed, <stream>, epoch)``.
    hard_start_jitter_s, hard_focal_gamma : float
        Replay of mined windows; see ``MemmapNoiseSampler``.
    device : str or torch.device
        Device of the batches.
    dtype : torch.dtype
        Dtype of the noise targets.

    Returns
    -------
    MemmapNoiseSampler or WhiteGaussianNoiseSampler
        Noise sampler.
    """
    source = noise[role]
    recolour = recoloured(noise, role)
    if source["name"] == "synthetic":
        return WhiteGaussianNoiseSampler(
            check_detector_asds(source["asd"], detectors), detectors, window, batch_size,
            stream_seed(seed, "noise", epoch), source["low_frequency_cutoff_in_hz"], source["filter_duration_in_s"], device, dtype,
        )
    runs = noise_runs(source, data_dir, detectors)
    postprocess = None
    if recolour:
        postprocess = build_recolour(
            noise["recolour"], runs, data_dir, window, detectors, stream_seed(seed, "recolour", epoch), device,
        )
    return MemmapNoiseSampler(
        runs, detectors, window, batch_size, noise_bad_segments([path for run in runs for path in run["bins"]]),
        stream_seed(seed, "noise", epoch), noise["prefetch_batches"], noise["read_workers"], hard_start_jitter_s, hard_focal_gamma,
        device, dtype, postprocess_fn=postprocess,
    )


def build_noise(
    noise: dict,
    data_dir: Path,
    window: Window,
    detectors: list,
    batch_size: int,
    seed: int,
    epoch: int,
    hard_start_jitter_s: float,
    hard_focal_gamma: float,
    device,
    dtype: torch.dtype,
):
    """
    Check the training and validation noise sources, then build the training noise sampler.

    Validation passes build their own stream each time (``build_noise_stream``).

    Parameters
    ----------
    noise, data_dir, window, detectors, batch_size, hard_start_jitter_s, hard_focal_gamma, device, dtype
        As ``build_noise_stream``.
    seed, epoch : int
        Seed of the training streams and the epoch the segment starts at.

    Returns
    -------
    MemmapNoiseSampler or WhiteGaussianNoiseSampler
        Training noise sampler.

    Raises
    ------
    ValueError
        If recolouring is asked of synthetic noise, or real training and
        validation sources share data.
    """
    for role in ("training", "validation"):
        recoloured(noise, role)
    if recoloured(noise, "validation"):
        logger.warning("noise.recolour.validation is on: validation noise is recoloured")
    if noise["training"]["name"] == noise["validation"]["name"] == "real":
        for training_run in noise_runs(noise["training"], data_dir, detectors):
            for validation_run in noise_runs(noise["validation"], data_dir, detectors):
                if training_run["run"] == validation_run["run"]:
                    check_disjoint(
                        selected_records(training_run, detectors),
                        selected_records(validation_run, detectors),
                        training_run["run"],
                    )
    return build_noise_stream(
        noise, "training", data_dir, window, detectors, batch_size, seed, epoch,
        hard_start_jitter_s, hard_focal_gamma, device, dtype,
    )


def build_preprocess(
    preprocess: list,
    asds: torch.Tensor,
    window: Window,
    bounds: dict,
    low_frequency_cutoff_in_hz: float,
    input_state: ProcessingState,
    device,
    dtype: torch.dtype,
) -> Preprocessor:
    """
    Build the preprocessing chain from the ``preprocess`` steps, in order.

    Parameters
    ----------
    preprocess : list of dict
        Checked ``preprocess`` section of a run file.
    asds : torch.Tensor, shape ``(D, F)``
        Fiducial ASDs, for whitening.
    window : Window
        Window geometry.
    bounds : dict
        Prior bounds from the signal sampler; ``mass1`` and ``tc`` set the
        multirate bins.
    low_frequency_cutoff_in_hz : float
        Signal low-frequency cutoff.
    input_state : ProcessingState
        State of the batches the chain receives.
    device : str or torch.device
        Device of the whitening and FIR kernels.
    dtype : torch.dtype
        Dtype of the FIR kernel.

    Returns
    -------
    Preprocessor
        The chain, its order checked.

    Raises
    ------
    ValueError
        If a step name is unknown or a multirate setting is invalid.
    PipelineError
        If a step cannot take the state the step before it gives.
    """
    steps = []
    for index, step in enumerate(preprocess):
        cls, settings = lookup(PREPROCESS_STEPS, step, f"preprocess[{index}]")
        if cls is FiducialWhitening:
            steps.append(FiducialWhitening(asds, window, device))
        elif cls is MinMaxNormalise:
            steps.append(MinMaxNormalise())
        else:
            binning = DyadicPyramidBinning(
                bounds["mass1"], bounds["tc"], window, low_frequency_cutoff_in_hz, **settings
            )
            steps.append(MultirateSampler(binning.detailed_bins, window, device, dtype))
    return Preprocessor(steps, input_state)


def resolve_component(table: dict, component: dict, provided: set, where: str) -> dict:
    """
    A named component's section with every parameter its class takes.

    Parameters
    ----------
    table : dict
        Component names to classes.
    component : dict
        Checked section: ``name`` and any parameters to set.
    provided : set of str
        ``__init__`` arguments the builder passes itself.
    where : str
        Section name, for error messages.

    Returns
    -------
    dict
        ``name``, then each parameter converted by the class's type hints;
        unset parameters take the class's defaults.

    Raises
    ------
    ValueError
        If the name or a parameter is unknown, or a value has the wrong type.
    """
    cls, parameters = lookup(table, component, where)
    arguments = component_arguments(cls, parameters, provided, where)
    signature = inspect.signature(cls.__init__).parameters
    defaults = {
        name: list(parameter.default) if isinstance(parameter.default, tuple) else parameter.default
        for name, parameter in signature.items()
        if name != "self" and name not in provided
    }
    return {"name": component["name"], **defaults, **arguments}


def resolve_settings(settings: dict) -> dict:
    """
    Checked settings with every named component's parameters filled in.

    The result is the run record: every value used, defaults included, and
    it reads back through ``load_run`` + ``resolve_settings`` unchanged.

    Parameters
    ----------
    settings : dict
        Settings from ``load_run``.

    Returns
    -------
    dict
        The same settings; ``signal.waveform``, ``signal.projection``,
        ``signal.snr``, ``model``, ``loss``, ``optimiser``, ``schedule`` and
        ``validation.snr`` resolved.
    """
    signal = settings["signal"]
    return {
        **settings,
        "signal": {
            **signal,
            "waveform": resolve_component(WAVEFORMS, signal["waveform"], WAVEFORM_PROVIDED, "signal.waveform"),
            "projection": resolve_component(
                PROJECTIONS, signal["projection"], PROJECTION_PROVIDED, "signal.projection"
            ),
            "snr": resolve_component(SNR_DISTRIBUTIONS, signal["snr"], SNR_PROVIDED, "signal.snr"),
        },
        "model": resolve_model(settings["model"]),
        "loss": resolve_loss(settings["loss"]),
        "optimiser": resolve_optimiser(settings["optimiser"]),
        "schedule": resolve_schedule(settings["schedule"]),
        "validation": {
            **settings["validation"],
            "snr": resolve_component(
                SNR_DISTRIBUTIONS, settings["validation"]["snr"], SNR_PROVIDED, "validation.snr"
            ),
        },
    }


def resolve_model(model: dict) -> dict:
    """The ``model`` section with every parameter; see :func:`resolve_component`."""
    return resolve_component(MODELS, model, MODEL_PROVIDED, "model")


def resolve_loss(loss: dict) -> dict:
    """The ``loss`` section with every parameter; see :func:`resolve_component`."""
    return resolve_component(LOSSES, loss, LOSS_PROVIDED, "loss")


def resolve_optimiser(optimiser: dict) -> dict:
    """
    The ``optimiser`` section with every parameter of the torch optimiser.

    Unset parameters take the production values for AdamW (``lr`` 2e-4,
    ``weight_decay`` 1e-4) and PyTorch's defaults otherwise; see
    :func:`resolve_component`.
    """
    settings = resolve_component(OPTIMISERS, optimiser, OPTIMISER_PROVIDED, "optimiser")
    production = OPTIMISER_DEFAULTS.get(settings["name"], {})
    return {**settings, **{key: value for key, value in production.items() if key not in optimiser}}


def check_heads(model: dict, loss: dict) -> None:
    """
    Check that the model's estimate heads give what the loss needs.

    Parameters
    ----------
    model, loss : dict
        Checked ``model`` and ``loss`` sections.

    Raises
    ------
    ValueError
        If the loss needs a width per estimate and the model has none, or
        the other way round; the message lists the names that would match.
    """
    model_cls, _ = lookup(MODELS, model, "model")
    loss_cls, _ = lookup(LOSSES, loss, "loss")
    if model_cls.uncertainty == loss_cls.uncertainty:
        return
    heads = {True: "an estimate and a width per target", False: "one estimate per target"}
    models = sorted(name for name, cls in MODELS.items() if cls.uncertainty == loss_cls.uncertainty)
    losses = sorted(name for name, cls in LOSSES.items() if cls.uncertainty == model_cls.uncertainty)
    raise ValueError(
        f"model {model['name']} gives {heads[model_cls.uncertainty]}, but loss {loss['name']} "
        f"needs {heads[loss_cls.uncertainty]}; use a loss from {losses} or a model from {models}"
    )


def build_model(
    model: dict, num_detectors: int, num_estimates: int, device, dtype: torch.dtype
) -> torch.nn.Module:
    """
    Build the network the ``model`` section names.

    Parameters
    ----------
    model : dict
        Checked ``model`` section.
    num_detectors : int
        Detector channels, from ``data.detectors``.
    num_estimates : int
        Estimated parameters, from ``signal.targets``.
    device : str or torch.device
        Device of the network.
    dtype : torch.dtype
        Dtype of the network.

    Returns
    -------
    torch.nn.Module
        Network in the default memory layout; training moves it to
        channels-last (which changes CPU results at rounding level).
    """
    settings = resolve_model(model)
    parameters = {key: value for key, value in settings.items() if key != "name"}
    network = MODELS[settings["name"]](num_detectors, num_estimates, **parameters)
    return network.to(device=device, dtype=dtype)


def build_loss(loss: dict, targets: list) -> torch.nn.Module:
    """
    Build the loss the ``loss`` section names.

    Parameters
    ----------
    loss : dict
        Checked ``loss`` section.
    targets : list of str
        Estimated parameters, from ``signal.targets``; name the components.

    Returns
    -------
    torch.nn.Module
        Loss returning its named components, total first.
    """
    settings = resolve_loss(loss)
    parameters = {key: value for key, value in settings.items() if key != "name"}
    return LOSSES[settings["name"]](list(targets), **parameters)


def build_optimiser(optimiser: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build the torch optimiser the ``optimiser`` section names.

    Weight decay applies only to parameters of two or more dimensions;
    normalisation scales and shifts and biases get none (He et al., "Bag of
    Tricks", CVPR 2019). The fused kernel is used whenever the optimiser offers
    one; PyTorch stops with its own error on a device that lacks it.

    Parameters
    ----------
    optimiser : dict
        Checked ``optimiser`` section.
    model : torch.nn.Module
        Network whose trainable parameters are optimised.

    Returns
    -------
    torch.optim.Optimizer
        Optimiser.
    """
    settings = resolve_optimiser(optimiser)
    cls = OPTIMISERS[settings["name"]]
    parameters = {key: value for key, value in settings.items() if key != "name"}
    signature = inspect.signature(cls.__init__).parameters

    trainable = [p for p in model.parameters() if p.requires_grad]
    if "weight_decay" in signature:
        groups = [
            {"params": [p for p in trainable if p.ndim >= 2], "weight_decay": parameters.pop("weight_decay")},
            {"params": [p for p in trainable if p.ndim < 2], "weight_decay": 0.0},
        ]
    else:
        groups = trainable
    if "fused" in signature:
        parameters["fused"] = True
    return cls(groups, **parameters)


def resolve_schedule(schedule: dict) -> dict:
    """
    The ``schedule`` section with every parameter of the schedule.

    ``WarmupPolynomialDecayFlatTail`` steps every batch and takes no
    ``step_every``; a PyTorch scheduler needs ``step_every`` (batch, epoch or
    metric), the unit its own parameters count in.

    Parameters
    ----------
    schedule : dict
        Checked ``schedule`` section.

    Returns
    -------
    dict
        ``name``, ``step_every`` (PyTorch schedulers only), then every
        parameter, defaults included.

    Raises
    ------
    ValueError
        On an unknown name or parameter, or a missing or misplaced ``step_every``.
    """
    cls, _ = lookup(SCHEDULES, schedule, "schedule")
    if cls is WarmupPolynomialDecayFlatTail:
        if "step_every" in schedule:
            raise ValueError(f"schedule.step_every: {schedule['name']} steps every batch; remove step_every")
        return resolve_component(SCHEDULES, schedule, SAGE_SCHEDULE_PROVIDED, "schedule")
    if "step_every" not in schedule:
        raise ValueError(f"schedule.step_every: required for {schedule['name']}: one of {', '.join(STEP_EVERY)}")
    step_every = convert(schedule["step_every"], STEP_EVERY, "schedule.step_every")
    parameters = {key: value for key, value in schedule.items() if key != "step_every"}
    settings = resolve_component(SCHEDULES, parameters, TORCH_SCHEDULE_PROVIDED, "schedule")
    return {"name": settings.pop("name"), "step_every": step_every, **settings}


def build_schedule(
    schedule: dict, optimiser: torch.optim.Optimizer, num_epochs: int, steps_per_epoch: int
) -> tuple:
    """
    Build the learning-rate schedule the ``schedule`` section names.

    Parameters
    ----------
    schedule : dict
        Checked ``schedule`` section.
    optimiser : torch.optim.Optimizer
        Optimiser it drives; its learning rate is the peak.
    num_epochs, steps_per_epoch : int
        Epochs and batches per epoch of the run.

    Returns
    -------
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Scheduler.
    step_every : str
        ``batch``, ``epoch`` or ``metric``.
    """
    settings = resolve_schedule(schedule)
    cls = SCHEDULES[settings["name"]]
    parameters = {key: value for key, value in settings.items() if key not in ("name", "step_every")}
    if cls is WarmupPolynomialDecayFlatTail:
        return cls(optimiser, num_epochs * steps_per_epoch, steps_per_epoch, **parameters), "batch"
    return cls(optimiser, **parameters), settings["step_every"]


def flat_tail_start_epoch(schedule: dict, num_epochs: int):
    """
    First epoch of the schedule's flat tail, or None if it has none.

    Parameters
    ----------
    schedule : dict
        Checked ``schedule`` section.
    num_epochs : int
        Epochs in the run.

    Returns
    -------
    int or None
        ``num_epochs - flat_tail_epochs`` for ``WarmupPolynomialDecayFlatTail``.
    """
    settings = resolve_schedule(schedule)
    if SCHEDULES[settings["name"]] is not WarmupPolynomialDecayFlatTail:
        return None
    return num_epochs - settings["flat_tail_epochs"]


def compile_arguments(compile_setting):
    """
    The ``train.compile`` setting as ``torch.compile`` arguments.

    Parameters
    ----------
    compile_setting : bool or dict
        ``False``, or torch.compile arguments with ``mode`` required.

    Returns
    -------
    dict or None
        Arguments, or None for no compilation.

    Raises
    ------
    ValueError
        On ``true`` without settings, a missing or unknown mode, or an
        argument torch.compile does not take.
    """
    if compile_setting is False:
        return None
    if compile_setting is True:
        raise ValueError("train.compile: give the torch.compile settings (at least mode), or false")
    arguments = component_arguments(torch.compile, compile_setting, {"model"}, "train.compile")
    if "mode" not in arguments:
        raise ValueError(f"train.compile.mode: required; one of {', '.join(COMPILE_MODES)}")
    if arguments["mode"] not in COMPILE_MODES:
        raise ValueError(
            f"train.compile.mode: {arguments['mode']!r} is not one of {', '.join(COMPILE_MODES)}"
        )
    return arguments


def check_train(train: dict) -> None:
    """
    Start-up checks of the ``train`` section.

    Parameters
    ----------
    train : dict
        Checked ``train`` section.

    Raises
    ------
    ValueError
        If an epoch is not whole batches, the signal fraction gives a
        fractional count, the compile settings are invalid, fewer than one
        checkpoint is kept, or the device is not a device name.
    """
    check_whole_batches("train.samples_per_epoch", train["samples_per_epoch"], train["batch_size"])
    check_whole_signals(train["signal_fraction"], train["batch_size"])
    compile_arguments(train["compile"])
    if train["keep_checkpoints"] != "all" and train["keep_checkpoints"] < 1:
        raise ValueError(f"train.keep_checkpoints {train['keep_checkpoints']} must be at least 1, or all")
    try:
        torch.device(train["device"])
    except RuntimeError as error:
        raise ValueError(f"train.device: {error}") from None


def check_device(train: dict) -> None:
    """
    Check that ``train.device`` and ``train.precision`` are available here.

    Parameters
    ----------
    train : dict
        ``train`` section passed by ``check_train``.

    Raises
    ------
    ValueError
        If the device is a GPU and none is visible, or the GPU lacks bf16.
    """
    if torch.device(train["device"]).type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"train.device {train['device']}: no GPU is visible here")
        if train["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("train.precision bf16: this GPU does not support it; use fp16 or fp32")


def check_validation(validation: dict, batch_size: int) -> None:
    """
    Start-up checks of the ``validation`` section.

    Parameters
    ----------
    validation : dict
        Checked ``validation`` section.
    batch_size : int
        ``train.batch_size``, which validation shares.

    Raises
    ------
    ValueError
        If a sample count is not whole batches, the final set is smaller than
        a regular pass, or ``every_n_epochs`` is below 1.
    """
    check_whole_batches("validation.samples", validation["samples"], batch_size)
    check_whole_batches("validation.final_samples", validation["final_samples"], batch_size)
    if validation["final_samples"] < validation["samples"]:
        raise ValueError(
            f"validation.final_samples {validation['final_samples']} is smaller than "
            f"validation.samples {validation['samples']}"
        )
    if validation["every_n_epochs"] < 1:
        raise ValueError(f"validation.every_n_epochs {validation['every_n_epochs']} must be at least 1")


def validates(epoch: int, validation: dict, epochs: int) -> bool:
    """Whether a regular validation pass follows ``epoch``: every
    ``every_n_epochs`` epochs from epoch 0, and the final epoch."""
    return epoch % validation["every_n_epochs"] == 0 or epoch == epochs - 1


def mining_epochs(mining: dict, schedule: dict, num_epochs: int) -> list:
    """Epochs after which a mining round runs: every ``every_n_epochs`` from
    ``start_epoch``, before the schedule's flat tail (or the run's end)."""
    stop = flat_tail_start_epoch(schedule, num_epochs)
    return list(range(mining["start_epoch"], num_epochs if stop is None else stop, mining["every_n_epochs"]))


def check_mining(mining: dict, noise: dict, model: dict, schedule: dict, num_epochs: int) -> None:
    """
    Start-up checks of an enabled ``mining`` section.

    Parameters
    ----------
    mining, noise, model, schedule : dict
        Checked sections of the run file.
    num_epochs : int
        Epochs in the run.

    Raises
    ------
    ValueError
        If the training noise is not real, the model has no frontend tap, the
        start epoch is negative, ``every_n_epochs`` is below 1, or no round
        falls before mining stops.
    ModuleNotFoundError
        If pyribs is not installed.
    """
    if noise["training"]["name"] != "real":
        raise ValueError(f"mining needs real training noise; noise.training is {noise['training']['name']}")
    if not issubclass(MODELS[model["name"]], FrontendTap):
        raise ValueError(f"mining needs a model with the frontend tap; {model['name']} has none")
    if mining["start_epoch"] < 0:
        raise ValueError(f"mining.start_epoch {mining['start_epoch']} must be at least 0")
    if mining["every_n_epochs"] < 1:
        raise ValueError(f"mining.every_n_epochs {mining['every_n_epochs']} must be at least 1")
    if not mining_epochs(mining, schedule, num_epochs):
        stop = flat_tail_start_epoch(schedule, num_epochs)
        raise ValueError(
            f"mining.start_epoch {mining['start_epoch']} leaves no mining round: mining stops at epoch "
            f"{num_epochs if stop is None else stop}"
        )
    if importlib.util.find_spec("ribs") is None:
        raise ModuleNotFoundError("mining needs pyribs: pip install 'ribs>=0.7'", name="ribs")


def precision_setup(train: dict) -> tuple:
    """
    Autocast dtype and loss scaler of ``train.precision``.

    Returns
    -------
    autocast_dtype : torch.dtype or None
        bf16 or fp16; None for fp32 (no autocast).
    scaler : torch.amp.GradScaler or None
        A loss scaler for fp16 only.
    """
    precision = train["precision"]
    scaler = torch.amp.GradScaler(torch.device(train["device"]).type) if precision == "fp16" else None
    return AUTOCAST_DTYPES[precision], scaler


def compile_network(network: torch.nn.Module, train: dict) -> torch.nn.Module:
    """The network compiled as ``train.compile`` says, or unchanged for ``false``."""
    arguments = compile_arguments(train["compile"])
    return network if arguments is None else torch.compile(network, **arguments)
