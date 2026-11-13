"""
Settings each run-file section accepts, and the type each is read as.

A spec is a type, a tuple of allowed strings, a one-element list (a list
of that spec), a ``(type, default)`` pair, a dict group or ``OneOf``;
:func:`sage.config.loader.check` gives the rules for group keys.
"""

from typing import Any

from sage.core.hardcode import _DETECTORS, PYCBC_FILTER_DURATION


class OneOf:
    """Spec met by the first of several specs that converts a value."""

    def __init__(self, *specs):
        self.specs = specs

    def __repr__(self) -> str:
        return " or ".join(describe(spec) for spec in self.specs)


class Null:
    """Spec of YAML ``null``: an optional argument left to its class."""

    def __repr__(self) -> str:
        return "null"


NULL = Null()


def describe(spec: Any) -> str:
    """
    A spec in words, for error messages.

    Parameters
    ----------
    spec : Any
        Spec from the schema.

    Returns
    -------
    str
    """
    if isinstance(spec, list):
        return f"a list of {describe(spec[0])}"
    if isinstance(spec, tuple):
        return f"one of {', '.join(spec)}"
    if isinstance(spec, dict):
        return "a group of settings"
    return getattr(spec, "__name__", repr(spec))


COMPONENT = {"name": str, "...": None}

DATA = {
    "detectors": [tuple(sorted(_DETECTORS))],
    "sample_rate_in_hz": float,
    "sample_length_in_s": float,
    "padding_length_in_s": float,
    "dtype": ("float32", "float64"),
}

# Shares of the non-astrophysical class: of the noise windows, of
# signal-in-one-detector pairs among them, and of arrival times inside
# the delta_tc band
NON_ASTROPHYSICAL = {
    "noise_share": float,
    "signal_noise_share": float,
    "in_band_share": float,
}

SIGNAL = {
    "prior": str,
    "low_frequency_cutoff_in_hz": float,
    "reference_frequency_in_hz": float,
    "waveform": COMPONENT,
    "projection": COMPONENT,
    "snr": COMPONENT,
    "targets": [str],
    "non_astrophysical?": NON_ASTROPHYSICAL,
}

REAL_SOURCE = {"name": str, "runs": [str], "portion?": [float], "gps_range?": [float]}

SYNTHETIC_SOURCE = {
    "name": str,
    "asd": {"...": None},
    "low_frequency_cutoff_in_hz": float,
    "filter_duration_in_s": (float, PYCBC_FILTER_DURATION),
}

SOURCE = {"name": {"real": REAL_SOURCE, "synthetic": SYNTHETIC_SOURCE}}

RECOLOUR = {
    "target": str,
    "probability": float,
    "asd_jitter_gain": float,
    "validation": bool,
}

NOISE = {
    "training": SOURCE,
    "validation": SOURCE,
    "recolour?": RECOLOUR,
    "fiducial": [str],
    "prefetch_batches": int,
    "read_workers": int,
}

MULTIRATE = {
    "name": str,
    "lowest_sample_rate_in_hz": float,
    "nyquist_margin_in_hz": float,
    "min_bin_length_in_s": float,
    "pre_merger_leeway_in_s": float,
    "post_merger_leeway_in_s": float,
}

# Steps, applied in the order listed
PREPROCESS = [
    {
        "name": {
            "whiten": {"name": str},
            "minmax_normalise": {"name": str},
            "multirate": MULTIRATE,
        }
    }
]

# A named network, loss, optimiser and schedule; parameters are checked
# against each __init__ (a PyTorch schedule also takes step_every)
MODEL = COMPONENT
LOSS = COMPONENT
OPTIMISER = COMPONENT
SCHEDULE = COMPONENT

TRAIN = {
    "epochs": int,
    "samples_per_epoch": int,
    "batch_size": int,
    "signal_fraction": float,
    "device": str,
    "precision": ("bf16", "fp16", "fp32"),
    "clip_norm": float,
    # false, or torch.compile arguments (mode required), checked against
    # torch.compile
    "compile": OneOf(bool, {"mode": str, "...": None}),
    "seed": int,
    "keep_checkpoints": OneOf(int, ("all",)),
}

VALIDATION = {
    "samples": int,
    "every_n_epochs": int,
    "final_samples": int,
    "seed": int,
    "snr": COMPONENT,
}

# `enabled: true` alone gives production mining; every other setting
# defaults to its production value
MINING = {
    "enabled": bool,
    "start_epoch": (int, 3),
    "every_n_epochs": (int, 4),
    "iterations": (int, 8000),
    "keep_threshold": (float, 1.0),
    "replay": {
        "probability": (float, 0.2),
        "replays_per_epoch": (float, 2.0),
        "focal_gamma": (float, 3.0),
        "start_jitter_in_s": (float, 2.0),
    },
    "archive": {
        "n_cells": (int, 2048),
        "descriptor_dim": (int, 8),
        "novelty_dist": (float, 0.15),
        "novelty_weight": (float, 1.5),
        "max_embeddings": (int, 50000),
    },
    "emitter": {
        "n_emitters": (int, 1),
        "batch_size": (int, 72),
        "learning_rate": (float, 0.1),
        "warmup": (int, 2048),
    },
}

# `enabled: true` alone gives production EMA
EMA = {
    "enabled": bool,
    "decay": (float, 0.9999),
    "calibration_batches": (int, 500),
}

SECTIONS = {
    "data": DATA,
    "signal": SIGNAL,
    "noise": NOISE,
    "preprocess": PREPROCESS,
    "model": MODEL,
    "loss": LOSS,
    "optimiser": OPTIMISER,
    "schedule": SCHEDULE,
    "train": TRAIN,
    "validation": VALIDATION,
    "mining": MINING,
    "ema": EMA,
}
