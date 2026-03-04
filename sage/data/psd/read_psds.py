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
    # Configs
    cfg = get_cfg()

    psds_all = []
    for det in ["H1", "L1"]:
        psd_dir = Path(cfg.export_dir) / "fiducial_psds"
        bin_path = psd_dir / f"fiducial_{det}_psd.bin"
        meta_path = psd_dir / f"fiducial_{det}_psd.json"

        with open(meta_path, "r") as f:
            meta = json.load(f)

        psds = np.fromfile(bin_path, dtype=np.float64)
        psds_all.append(psds)

    return torch.from_numpy(np.stack(psds_all, axis=0)).to(device=cfg.device)
