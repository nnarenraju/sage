"""Load a trained network and what a search needs to run it.

The network, its training configs, the whitening and multirate
preprocessor, and the constants that decode its point estimates.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Type

import numpy as np
import torch

from sage.core.config import register_configs
from sage.data.waveform import read_from_config
from sage.factory.contract import make_processor


def as_namespace(config: dict) -> SimpleNamespace:
    """Config dict as attributes, with a stored dtype name made a dtype.

    Parameters
    ----------
    config : dict
        Config recorded in a checkpoint.

    Returns
    -------
    types.SimpleNamespace
        The same values, readable as attributes.
    """
    values = dict(config)
    if isinstance(values.get("dtype"), str):
        values["dtype"] = getattr(torch, values["dtype"].split(".")[-1])
    return SimpleNamespace(**values)


def load_network(
    path: Path, architecture: Type[torch.nn.Module], device: str = "cuda"
) -> Tuple[torch.nn.Module, dict, dict]:
    """Build a network and load its trained weights.

    Constructor arguments of ``architecture`` are read from the training
    config by name. The configs are registered globally, with ``device``
    set to the one requested, since other modules read them from there.

    Parameters
    ----------
    path : Path
        Checkpoint file.
    architecture : type
        Network class the checkpoint was trained as.
    device : str
        Device to load the network onto.

    Returns
    -------
    model : torch.nn.Module
        Network in evaluation mode.
    cfg, data_cfg : dict
        Training and data configs recorded in the checkpoint.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(checkpoint["cfg"], device=device)
    data_cfg = checkpoint["data_cfg"]
    register_configs(as_namespace(cfg), as_namespace(data_cfg))

    parameters = inspect.signature(architecture).parameters
    model = architecture(**{name: cfg[name] for name in parameters if name in cfg})

    # torch.compile prefixes every weight name with "_orig_mod.".
    state = {
        name.removeprefix("_orig_mod."): tensor
        for name, tensor in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state)
    model = model.to(device=device, dtype=torch.float32)
    return model.eval(), cfg, data_cfg


def encoding(
    param_sampler: object, targets: List[str], minmax: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Offset and scale each target was encoded with during training.

    Parameters
    ----------
    param_sampler : DistributionSampler
        Training parameter sampler holding the encoding buffers.
    targets : list of str
        Point-estimate targets in head order, ``cfg.do_point_estimate``.
    minmax : bool
        Whether training used min-max encoding, ``cfg.pe_target_minmax``.

    Returns
    -------
    offsets, scales : numpy.ndarray
        Encoding constants per target, as float64.
    """
    names = ("_norm_mins", "_norm_scales", "_norm_indices")
    if not minmax:
        names = ("_std_means", "_std_stds", "_std_indices")
    offsets, scales, indices = (
        getattr(param_sampler, name).detach().cpu().numpy().ravel() for name in names
    )

    columns = indices.tolist()
    positions = [columns.index(param_sampler.param_index[name]) for name in targets]
    return offsets[positions].astype(np.float64), scales[positions].astype(np.float64)


def load_search_network(
    checkpoint: Path,
    architecture: Type[torch.nn.Module],
    gwconfig: Path,
    device: str = "cuda",
    sampler_seed: int = 150914,
) -> Dict[str, object]:
    """Network, preprocessor and decoding constants for a search.

    The parameter sampler is the training one, built from the training
    ``gwconfig.yaml``. It sets the multirate binning and the encoding
    the point estimates are decoded with.

    Parameters
    ----------
    checkpoint : Path
        Training checkpoint.
    architecture : type
        Network class the checkpoint was trained as.
    gwconfig : Path
        Parameter prior the network was trained under.
    device : str
        Device to score on.
    sampler_seed : int
        Seed of the sampler; its encoding constants are Monte Carlo
        estimates.

    Returns
    -------
    dict of str to object
        ``model``, ``processor``, ``detectors``, ``window_samples``,
        ``sample_rate``, ``padding_s``, ``targets``, ``offsets``, ``scales``,
        and the ``provenance`` written into every trigger file.
    """
    model, cfg, data_cfg = load_network(checkpoint, architecture, device)

    sampler = read_from_config(str(gwconfig), seed=sampler_seed)
    # Training compiles these inside the waveform generator, which a
    # search does not build.
    sampler._compile_batch_normaliser()
    sampler._compile_batch_standardiser()

    targets = list(cfg["do_point_estimate"])
    offsets, scales = encoding(sampler, targets, bool(cfg.get("pe_target_minmax")))
    sample_rate = data_cfg["sample_rate"]
    padding_s = data_cfg["padding_length_in_s"]
    window_s = data_cfg["sample_length_in_s"] + 2 * padding_s

    return {
        "model": model,
        "processor": make_processor(sampler.bounds),
        "detectors": list(cfg["detectors"]),
        "window_samples": round(window_s * sample_rate),
        "sample_rate": sample_rate,
        "padding_s": padding_s,
        "targets": targets,
        "offsets": offsets,
        "scales": scales,
        "provenance": {
            "checkpoint": str(Path(checkpoint).resolve()),
            "architecture": architecture.__name__,
            "gwconfig": str(Path(gwconfig).resolve()),
            "sampler_seed": sampler_seed,
        },
    }
