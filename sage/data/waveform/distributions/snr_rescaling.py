#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : snr_rescaling.py
Description     : Short description of the file

Created on 2026-03-10 03:59:41

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
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
import torch
import torch.nn as nn


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
        Scale parameter of the half-normal.
    loc : float
        Location shift added after folding.
    seed : int
        Seed for the internal :class:`torch.Generator`.
    device : str or torch.device
        Device of the draws.
    """

    def __init__(self, scale: float, loc: float, seed: int, device):
        super().__init__()

        self.register_buffer("scale", torch.tensor(scale).to(device=device))
        self.register_buffer("loc", torch.tensor(loc).to(device=device))

        # Create a generator with a specific seed
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

        # Most recent draw, read by SageVanillaTesting.run_signal.
        self.last = None

    def forward(self, batch_size: int):
        x = torch.randn(
            batch_size,
            device=self.scale.device,
            generator=self.gen,
        ).abs()
        self.last = x * self.scale + self.loc
        return self.last
