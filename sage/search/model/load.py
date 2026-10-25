"""Load a trained network and what a search needs to run it, from one checkpoint.

The checkpoint's record gives the network, the preprocessor (fiducial ASDs
included) and the target scaling that decodes the point estimates.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml

from sage.config.loader import load_run_record, read_yaml_text
from sage.config.prior import check_prior
from sage.core.pipeline import Grid, ProcessingState
from sage.data.asd import fiducial_asds_from_record
from sage.data.waveform import prior_sampler
from sage.dsp.multirate_sampling import MultirateSampler
from sage.factory.build import build_data, build_model, build_preprocess, resolve_loss


def load_network(path: Path, device: str = "cuda") -> Tuple[torch.nn.Module, dict, dict]:
    """Build the network a checkpoint names and load its weights.

    Parameters
    ----------
    path : Path
        Checkpoint file (``latest.pt``, ``best.pt``, ``ema.pt``, ...).
    device : str
        Device to load the network onto.

    Returns
    -------
    model : torch.nn.Module
        Network in evaluation mode.
    settings : dict
        The run settings the checkpoint records.
    record : dict
        The checkpoint's run record.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    record = checkpoint["record"]
    settings = load_run_record(record["run"])
    model = build_model(
        settings["model"], len(settings["data"]["detectors"]), len(settings["signal"]["targets"]),
        device, torch.float32,
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval(), settings, record


def load_search_network(checkpoint: Path, device: str = "cuda") -> Dict[str, object]:
    """Network, preprocessor and decoding constants for a search.

    Parameters
    ----------
    checkpoint : Path
        Training checkpoint.
    device : str
        Device to score on.

    Returns
    -------
    dict of str to object
        ``model``, ``processor``, ``detectors``, ``window_samples``,
        ``sample_rate``, ``padding_s``, ``targets``, ``offsets``, ``scales``,
        ``width_bounds``, and the ``provenance`` written into every trigger file.

    Raises
    ------
    ValueError
        If the multirate bins rebuilt from the record differ from those the
        network was trained with.
    """
    model, settings, record = load_network(checkpoint, device)
    window, detectors, dtype = build_data(settings["data"])
    prior = check_prior(read_yaml_text(record["prior"], "checkpoint record prior.yaml"))
    derived = yaml.safe_load(record["derived"])

    # Bounds are theoretical, so the sampler's seed does not enter them.
    bounds = prior_sampler(prior, 0, device, dtype).bounds
    asds = fiducial_asds_from_record(record["fiducial"], detectors, window, device)
    processor = build_preprocess(
        settings["preprocess"], asds, window, bounds,
        settings["signal"]["low_frequency_cutoff_in_hz"], ProcessingState(Grid.FD_UNIFORM), device, dtype,
    )
    bins = [step.bins.tolist() for step in processor.seq if isinstance(step, MultirateSampler)]
    if bins != derived["multirate_bins"]:
        raise ValueError(
            f"{checkpoint}: multirate bins rebuilt from the record differ from training's; "
            "the binning code has changed since this network was trained"
        )

    targets = list(settings["signal"]["targets"])
    scaling = derived["target_scaling"]
    loss = resolve_loss(settings["loss"])
    return {
        "model": model,
        "processor": processor,
        "detectors": detectors,
        "window_samples": window.padded_length_in_samples,
        "sample_rate": window.sample_rate_in_hz,
        "padding_s": window.padding_length_in_s,
        "targets": targets,
        "offsets": np.array([scaling[target]["offset"] for target in targets], dtype=np.float64),
        "scales": np.array([scaling[target]["scale"] for target in targets], dtype=np.float64),
        "width_bounds": (loss["sigma_min"], loss["sigma_max"]),
        "provenance": {
            "checkpoint": str(Path(checkpoint).resolve()),
            "model": settings["model"]["name"],
        },
    }
