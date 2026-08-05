#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : read_asds.py
Description     : Load the run's pre-computed fiducial ASDs from disk.

Created on 2026-03-02 20:46:58

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import json
import torch
import numpy as np

from pathlib import Path

# LOCAL
from sage.core.config import get_cfg


def get_fiducial_asds():
    """
    Load the pre-computed fiducial per-detector ASDs from disk.

    Reads binary float32 files written during the data-preparation stage
    (:class:`~sage.data.primer.get_asds.EstimateASD`) and returns them as a
    single stacked tensor on the configured device.

    These are amplitude spectral densities in strain/sqrt(Hz) (~1e-23), not
    power spectral densities (~1e-46) -- which is why
    :class:`~sage.dsp.whiten.FiducialWhitening` divides strain by them directly
    and :class:`~sage.data.waveform.snr.OptimalSNREstimator` divides once rather
    than by a square root.  The ``fiducial_{det}_psd.bin`` filenames are
    historical and are kept so existing run exports stay readable.

    Returns
    -------
    torch.Tensor, shape ``(D, F)``
        Per-detector one-sided ASDs on ``cfg.device``, where ``D`` is the
        number of detectors and ``F`` is the number of frequency bins.
    """
    # Configs
    cfg = get_cfg()

    # Fiducial ASDs are per-detector and shared across a run's detector-set
    # networks (HL/LV/HV/HLV), so a config may point ``fiducial_dir`` at a shared
    # location instead of the per-network ``export_dir``. Defaults to the old
    # ``{export_dir}/fiducial_psds`` when unset.
    asd_dir = Path(getattr(cfg, "fiducial_dir", None)
                   or (Path(cfg.export_dir) / "fiducial_psds"))

    asds_all = []
    for det in cfg.detectors:
        bin_path = asd_dir / f"fiducial_{det}_psd.bin"
        meta_path = asd_dir / f"fiducial_{det}_psd.json"

        with open(meta_path, "r") as f:
            meta = json.load(f)

        asds = np.fromfile(bin_path, dtype=np.float32)
        asds_all.append(asds)

    return torch.from_numpy(np.stack(asds_all, axis=0)).to(device=cfg.device)
