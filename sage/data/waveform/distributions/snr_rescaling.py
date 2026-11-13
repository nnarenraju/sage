#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Half-normal distribution of target network SNRs.

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

import torch
import torch.nn as nn


class HalfNorm(nn.Module):
    """
    Target network SNRs drawn as ``loc + scale * |N(0, 1)|``.

    The ``target_snr_sampler`` of
    :class:`~sage.data.waveform.snr.OptimalSNRRescaler`.

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

        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

        # Most recent draw, read by SageVanillaTesting.run_signal.
        self.last = None

    def forward(self, batch_size: int):
        """
        Draw ``batch_size`` target SNRs and keep them in ``last``.

        Parameters
        ----------
        batch_size : int
            Number of draws.

        Returns
        -------
        torch.Tensor, shape ``(batch_size,)``
            Target SNRs.
        """
        x = torch.randn(
            batch_size,
            device=self.scale.device,
            generator=self.gen,
        ).abs()
        self.last = x * self.scale + self.loc
        return self.last
