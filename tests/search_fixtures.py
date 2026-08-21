#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : search_fixtures.py
Description   : Synthetic checkpoint and strain release for testing the search offline.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Most of the search can be verified without a GPU and without the 1.3 TB of strain on
project storage, but only if the two things it reads can be fabricated faithfully. These
build them.

Faithfully is the operative word. A fixture that is *tidier* than reality tests nothing:
a contiguous, GPS-sorted release would pass a reader that assumes both, and the real
release is neither. So these deliberately reproduce the awkward properties:

* chunks overlap, and the overlapping samples **differ** between the two chunks that hold
  them, so splicing across a boundary is detectable rather than merely wrong;
* ``segment_index`` is positional and **anti-correlated** with GPS, as it is in the real
  sidecars, where records are numbered in parallel-completion order;
* with the default fill, every sample in a chunk carries that chunk's own value, so any
  window that crosses a boundary has non-zero peak-to-peak and is caught immediately;
* the network fixture is built in both norm flavours, because a separability test with no
  negative control passes on any model whose frontend happens to be an identity.

Nothing here writes outside the directory it is given.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# The scale factor the real releases store strain in; the reader divides it out.
DYN_RANGE_FAC = 5.902958103587057e20

# Measured from the real O3a and O3b sidecars: consecutive chunks overlap by this much,
# which is one analysis window short of two trims and is why a boundary band can host no
# window start.
REAL_OVERLAP_S = 15.5994

DEFAULT_SAMPLE_RATE = 2048.0
DEFAULT_CHUNK_S = 512.0


# ------------------------------------------------------------------ strain release
def make_synthetic_release(
    root: str | Path,
    detectors: Sequence[str] = ("H1", "L1"),
    observing_run: str = "O3a",
    n_chunks: int = 4,
    chunk_s: float = DEFAULT_CHUNK_S,
    overlap_s: float = REAL_OVERLAP_S,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    gps_start: float = 1238166018.0,
    fill: str = "constant",
    shuffle_index: bool = True,
    seed: int = 20260809,
) -> Path:
    """
    Write a miniature strain release in the real on-disk format.

    Parameters
    ----------
    fill : {"constant", "noise"}
        ``"constant"`` gives every sample of a chunk that chunk's own positional index
        plus one, so a window crossing a chunk boundary has non-zero peak-to-peak and a
        splicing bug is caught by a single assertion. ``"noise"`` draws each chunk
        independently, so overlapping regions hold genuinely different data.
    shuffle_index : bool
        Number the records in an order anti-correlated with GPS, as the real releases do.
        Setting this false produces a GPS-sorted release, which is useful only to confirm
        that a test would have passed for the wrong reason.

    Returns
    -------
    Path
        The release directory.

    Notes
    -----
    The layout matches what :mod:`sage.data.noise.real_noise` reads: a flat little-endian
    float32 ``.bin`` per detector and a JSON sidecar whose records carry ``sample_start_idx``
    and ``nsamples`` into it. Records are contiguous in the file and shuffled in time.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    step_s = chunk_s - overlap_s
    n_samples = int(round(chunk_s * sample_rate))

    for detector in detectors:
        # Chunks in true time order, then numbered in a different order below.
        spans = [
            (gps_start + k * step_s, gps_start + k * step_s + chunk_s)
            for k in range(n_chunks)
        ]
        order = list(range(n_chunks))
        if shuffle_index:
            order = order[::-1]

        bin_path = root / f"data_{detector}_{observing_run}.bin"
        records = []
        cursor_samples = 0
        cursor_bytes = 0
        with open(bin_path, "wb") as fh:
            for position, chunk in enumerate(order):
                if fill == "constant":
                    values = np.full(n_samples, float(position + 1), dtype=np.float64)
                elif fill == "noise":
                    values = rng.standard_normal(n_samples)
                else:
                    raise ValueError(f"unknown fill {fill!r}")
                stored = (values * DYN_RANGE_FAC).astype("<f4")
                payload = stored.tobytes()
                fh.write(payload)
                records.append(
                    {
                        "segment_index": position,
                        "detector": detector,
                        "observing_run": observing_run,
                        "gps_start": spans[chunk][0],
                        "gps_end": spans[chunk][1],
                        "sample_rate": sample_rate,
                        "nsamples": n_samples,
                        "dtype": "float32",
                        "endianness": "<",
                        "sample_start_idx": cursor_samples,
                        "byte_offset": cursor_bytes,
                        "byte_length": len(payload),
                        "checksum": hashlib.sha256(payload).hexdigest(),
                        "checksum_algorithm": "sha256",
                        "dyn_range_fac": DYN_RANGE_FAC,
                        "noise_low_freq_cutoff": 15.0,
                    }
                )
                cursor_samples += n_samples
                cursor_bytes += len(payload)

        sidecar = root / f"data_{detector}_{observing_run}_segments.json"
        sidecar.write_text(json.dumps(records, indent=2), encoding="utf-8")
        (root / f"data_{detector}_{observing_run}_failed_segments.json").write_text(
            "[]", encoding="utf-8"
        )

    return root


def release_is_gps_sorted(root: str | Path, detector: str, observing_run: str) -> bool:
    """Whether a release happens to be in time order, for asserting a fixture is not."""
    path = Path(root) / f"data_{detector}_{observing_run}_segments.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    starts = [r["gps_start"] for r in records]
    return all(b > a for a, b in zip(starts, starts[1:]))


# ------------------------------------------------------------------ network fixture
class SharedScaleNorm(nn.Module):
    """
    Normalisation that scales every channel by a statistic pooled over all detectors.

    A negative control for :meth:`~sage.search.network.SplitNetwork.separability` that a
    shift-based probe cannot see. The coupling is real -- every detector's frontend input
    depends on every detector's samples -- but it enters through a standard deviation,
    which is invariant under adding a constant to a channel. Only a probe that replaces a
    channel's samples, as a time slide does, moves it.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Divide by the pooled standard deviation, keeping the detector axis."""
        return x / (x.std(dim=(1, 2), keepdim=True) + 1e-5)


class ToyFrontendNet(nn.Module):
    """
    A miniature network with the same detector-coupling structure as the real one.

    Mirrors the ordering that decides separability: the normalisation is applied to the
    whole input first, then a per-detector frontend runs on one channel each, and only
    then are the channels concatenated for the backend.

    Under ``instancenorm`` the normalisation is per channel, so a detector's frontend
    output depends on that detector alone and features may be cached across time slides.
    Under ``groupnorm`` a single group spans the detector axis, every output depends on
    every input, and caching is invalid. ``sharedscale`` is the same invalidity reached
    through a shift-invariant statistic, which a probe that only offsets a channel cannot
    detect. All three are built here so the check has negative controls of both kinds.
    """

    def __init__(
        self,
        num_detectors: int = 2,
        norm_type: str = "instancenorm",
        length: int = 64,
        channels: int = 4,
        num_pe: int = 2,
    ):
        super().__init__()
        self.num_detectors = int(num_detectors)
        self.norm_type = str(norm_type)
        self.num_pe = int(num_pe)

        if norm_type == "instancenorm":
            self.norm = nn.InstanceNorm1d(num_detectors)
        elif norm_type == "groupnorm":
            self.norm = nn.GroupNorm(1, num_detectors)
        elif norm_type == "sharedscale":
            self.norm = SharedScaleNorm()
        else:
            raise ValueError(f"unknown norm_type {norm_type!r}")

        self.frontend = nn.ModuleList(
            [nn.Conv1d(1, channels, kernel_size=3, padding=1) for _ in range(num_detectors)]
        )
        self.backend = nn.Conv1d(channels * num_detectors, channels, kernel_size=1)
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.get_ranking_statistic = nn.Linear(channels, 1)
        self.point_estimate_layers = nn.ModuleList(
            [nn.Linear(channels, 2) for _ in range(num_pe)]
        )

    def forward_frontend(self, x: torch.Tensor, detector: int) -> torch.Tensor:
        """
        One detector's frontend output.

        Exposed separately because that is exactly what a separability check has to
        compare, and what a feature cache would store.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            normed = self.norm(x.float())
        return self.frontend[detector](normed[:, detector : detector + 1])

    def forward(self, x: torch.Tensor):
        """``(B, D, L)`` in; ``(ranking_statistic, point_estimates)`` out."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            normed = self.norm(x.float())
        cnn_outputs = [
            layer(normed[:, i : i + 1]) for i, layer in enumerate(self.frontend)
        ]
        features = self.backend(torch.cat(cnn_outputs, dim=1))
        features = self.flatten(self.avg_pool_1d(features))
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.float()
            ranking_statistic = self.get_ranking_statistic(features)
            raw = [layer(features) for layer in self.point_estimate_layers]
            mus = torch.cat([r[:, :1] for r in raw], dim=1)
            sigma_raw = torch.cat([r[:, 1:] for r in raw], dim=1)
            point_estimates = torch.cat([mus, sigma_raw], dim=1)
        return ranking_statistic, point_estimates


def make_synthetic_fiducial(
    root: str | Path,
    detectors: Sequence[str] = ("H1", "L1"),
    num_freq_bins: int = 16385,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    delta_f: float = 0.0625,
) -> Path:
    """
    Write fiducial ASDs in the on-disk layout ``FiducialWhitening`` reads.

    Amplitude spectral densities in strain/sqrt(Hz), not power -- the ``_psd.bin``
    filename is historical and is matched here so the whitener finds them. The shape is a
    smooth power law with a low-frequency wall, which is enough to exercise the whitening
    path; it is not a detector model and no sensitivity number may be taken from it.

    Returns
    -------
    Path
        The directory, ready to be a ``fiducial_dir``.
    """
    import json

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    freqs = np.arange(num_freq_bins, dtype=np.float64) * delta_f
    for detector in detectors:
        # A bucket-shaped ASD: a steep seismic wall below 15 Hz, a floor near 100 Hz and
        # a gentle shot-noise rise above it. Clipped away from zero so dividing by it is
        # defined at DC.
        shape = 1e-23 * (
            1.0 + (15.0 / np.clip(freqs, 1.0, None)) ** 6 + (freqs / 500.0) ** 2
        )
        asd = shape.astype(np.float32)
        asd.tofile(root / f"fiducial_{detector}_psd.bin")
        (root / f"fiducial_{detector}_psd.json").write_text(
            json.dumps(
                {
                    "detector": detector,
                    "num_freq_bins": int(num_freq_bins),
                    "dtype": "float32",
                    "byte_order": "little",
                    "sample_rate": float(sample_rate),
                    "delta_f": float(delta_f),
                    "freq_start": 0.0,
                    "freq_end": float(freqs[-1]),
                    "psd_aggregation": "synthetic",
                }
            )
        )
    return root


def make_synthetic_checkpoint(
    path: str | Path,
    detectors: Sequence[str] = ("H1", "L1"),
    norm_type: str = "instancenorm",
    observing_run: str = "O3b",
    fiducial_dir: str = "/nonexistent/fiducial_psds_o3ab",
    epoch: int = 7,
    val_loss: float = 0.2275,
    seed: int = 20260809,
    length: int = 64,
    extra_cfg: Optional[Dict] = None,
    extra_data_cfg: Optional[Dict] = None,
) -> Path:
    """
    Write a checkpoint in the current on-disk format, with flat configuration dicts.

    The format matters as much as the weights. A checkpoint that stores a *pickled config
    object* rather than a dict cannot be reopened once the class is refactored away, which
    is exactly what happened to every checkpoint written before the flat-dict change. The
    loader has to reject that case clearly, so this writes the good format and
    :func:`make_legacy_checkpoint` writes the bad one.

    Returns
    -------
    Path
        The checkpoint file.
    """
    torch.manual_seed(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = ToyFrontendNet(num_detectors=len(detectors), norm_type=norm_type, length=length)

    cfg = {
        "detectors": list(detectors),
        "norm_type": norm_type,
        "fiducial_dir": fiducial_dir,
        "do_point_estimate": ["tc", "mchirp"],
        "train_runs": [observing_run],
        "autocast": True,
        "dtype": "torch.float32",
        "device": "cuda:0",
        "batch_size": 64,
        "pe_target_minmax": False,
        "export_dir": str(path.parent.parent),
    }
    cfg.update(extra_cfg or {})

    data_cfg = {
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "sample_length_in_s": 12.0,
        "padding_length_in_s": 2.0,
        "padded_length_in_s": 16.0,
        "padded_length_in_nsamples": 32768,
        "padding_nsamples": 4096,
        "padded_delta_f": 0.0625,
        "delta_f": 1.0 / 12.0,
        "noise_low_frequency_cutoff": 15.0,
        "signal_low_frequency_cutoff": 20.0,
        "detectors": list(detectors),
    }
    data_cfg.update(extra_data_cfg or {})

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
            "data_cfg": data_cfg,
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )
    return path


class _UnpicklableConfig:
    """Stands in for a config class that no longer exists when the checkpoint is reopened."""


def make_legacy_checkpoint(path: str | Path, detectors: Sequence[str] = ("H1", "L1")) -> Path:
    """
    Write a checkpoint in the superseded format, storing a config *object*.

    Reproduces the failure that makes every pre-refactor checkpoint unreadable, so the
    loader's refusal can be tested rather than assumed. The object is picklable here and
    the point is only that ``cfg`` is not a dict of primitives; the loader must reject it
    on that basis rather than by attempting the load and crashing inside pickle.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = ToyFrontendNet(num_detectors=len(detectors))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": _UnpicklableConfig(),
            "data_cfg": _UnpicklableConfig(),
            "epoch": 127,
        },
        path,
    )
    return path


def toy_batch(
    n_detectors: int = 2, batch: int = 3, length: int = 64, seed: int = 11
) -> torch.Tensor:
    """A reproducible time-domain batch shaped ``(B, D, L)``."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n_detectors, length, generator=generator)
