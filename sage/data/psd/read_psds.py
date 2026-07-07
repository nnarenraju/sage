#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : read_psds.py
Description     : Short description of the file

Created on 2026-03-02 20:46:58

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = MIT Licence
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


def get_fiducial_psds():
    """
    Load the pre-computed fiducial per-detector PSDs from disk.

    Reads binary float32 files written during the data-preparation stage
    from ``{export_dir}/fiducial_psds/`` and returns them as a single
    stacked tensor on the configured device.

    The fiducial PSDs are used by
    :class:`~sage.data.waveform.snr.OptimalSNREstimator` to compute
    matched-filter SNR for SNR rescaling during signal injection.

    Returns
    -------
    torch.Tensor, shape ``(D, F)``
        Per-detector one-sided PSDs on ``cfg.device``, where ``D`` is the
        number of detectors and ``F`` is the number of frequency bins.
    """
    # Configs
    cfg = get_cfg()

    # Fiducial PSDs are per-detector and shared across a run's detector-set
    # networks (HL/LV/HV/HLV), so a config may point ``fiducial_dir`` at a shared
    # location instead of the per-network ``export_dir``. Defaults to the old
    # ``{export_dir}/fiducial_psds`` when unset.
    psd_dir = Path(getattr(cfg, "fiducial_dir", None)
                   or (Path(cfg.export_dir) / "fiducial_psds"))

    psds_all = []
    for det in cfg.detectors:
        bin_path = psd_dir / f"fiducial_{det}_psd.bin"
        meta_path = psd_dir / f"fiducial_{det}_psd.json"

        with open(meta_path, "r") as f:
            meta = json.load(f)

        psds = np.fromfile(bin_path, dtype=np.float32)
        psds_all.append(psds)

    return torch.from_numpy(np.stack(psds_all, axis=0)).to(device=cfg.device)
