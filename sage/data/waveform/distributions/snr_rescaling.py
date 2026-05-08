#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : snr_rescaling.py
Description     : Short description of the file

Created on 2026-03-10 03:59:41

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
import torch
import torch.nn as nn

# LOCAL
from sage.core.config import get_cfg


class HalfNorm(nn.Module):
    """
    Half-normal SNR sampler.

    Draws target network SNR values from a half-normal distribution
    ``|N(loc, scale²)|``.  Used as the ``target_snr_sampler`` argument to
    :class:`~sage.data.waveform.snr.OptimalSNRRescaler`.

    The generator is seeded once at construction so SNR draws are
    reproducible across runs with the same seed.

    Parameters
    ----------
    scale : float
        Scale parameter of the half-normal (default ``1.0``).
    loc : float
        Location shift added after folding (default ``0.0``).
    seed : int or None
        Seed for the internal :class:`torch.Generator`.
    """

    def __init__(self, scale=1.0, loc=0.0, seed=None):
        super().__init__()

        # Shared config
        cfg = get_cfg()

        self.register_buffer("scale", torch.tensor(scale).to(device=cfg.device))
        self.register_buffer("loc", torch.tensor(loc).to(device=cfg.device))

        # Create a generator with a specific seed
        self.gen = torch.Generator(device=cfg.device)
        self.gen.manual_seed(seed)

    def forward(self, batch_size: int):
        x = torch.randn(
            batch_size,
            device=self.scale.device,
            generator=self.gen,
        ).abs()
        return x * self.scale + self.loc
