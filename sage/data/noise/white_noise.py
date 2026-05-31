#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : white_noise.py
Description   : White Gaussian noise generators for pipeline testing.

Created on 2026-01-19 16:18:49

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import numpy as np
import torch

# LOCAL
from sage.core.config import get_cfg, get_data_cfg


class WhiteNoiseGenerator:
    """
    Generate independent white Gaussian noise for each detector.

    Produces zero-mean, unit-variance Gaussian noise with independent seeds
    per detector.  Primarily used for controlled testing and as a substitute
    for real noise during pipeline development or unit tests.
    """

    def generate(self, sample_length_in_num, seed=0):
        """
        Draw a single white Gaussian noise realisation.

        Parameters
        ----------
        sample_length_in_num : int
            Number of samples to generate.
        seed : int
            NumPy random seed for reproducibility.

        Returns
        -------
        numpy.ndarray, shape ``(sample_length_in_num,)``
            Zero-mean, unit-variance Gaussian noise.
        """
        np.random.seed(seed)
        # 0 mean, 1 std
        return np.random.normal(0, 1, size=sample_length_in_num)

    def apply(self, special, det_only=""):
        """
        Generate dual-detector white noise for a single sample.

        Parameters
        ----------
        special : dict
            Must contain ``"sample_seed"`` (int) and ``"data_cfg"`` with
            ``signal_length`` (s) and ``sample_rate`` (Hz) attributes.
        det_only : str
            Unused; kept for API compatibility.

        Returns
        -------
        numpy.ndarray, shape ``(2, N)``
            Stacked H1/L1 white noise arrays.
        """
        # Generate white Gaussian noise using random seeds
        rs = np.random.RandomState(seed=special["sample_seed"])
        seeds = list(rs.randint(0, 2**32, 2))  # one for each detector
        # Get sample length in num
        sample_length_in_s = special["data_cfg"].signal_length  # in seconds
        sample_rate = special["data_cfg"].sample_rate  # in samples/second
        sample_length_in_num = int(sample_length_in_s * sample_rate)
        # Generate noise for each detector
        H1_noise = self.generate(sample_length_in_num, seeds[0])
        L1_noise = self.generate(sample_length_in_num, seeds[1])
        # Return noise to dataset object
        noise = np.stack([H1_noise, L1_noise], axis=0)
        return noise


class WhiteGaussianNoiseSampler(torch.nn.Module):
    """
    Batch white Gaussian noise sampler for pipeline development and testing.

    Generates independent zero-mean unit-variance Gaussian noise in the time
    domain per detector, converts to the frequency domain via rfft
    (``norm='forward'``), and returns a GPU-resident FD_UNIFORM batch that
    mirrors the :class:`~sage.data.noise.real_noise.MemmapNoiseSampler` API.

    The returned noise is ``(B, D, F)`` complex, ready to be combined with
    signal batches and passed through :class:`~sage.dsp.whiten.FiducialWhitening`.
    When the signal sampler uses worst-case multibanding, the training loop's
    auto-multibanding selector converts this to FD_COARSE automatically before
    injection.

    Parameters
    ----------
    seed : int or None
        Seed for the internal NumPy RNG (for reproducibility).

    Attributes
    ----------
    GRAPH_READY : bool
        ``False`` — ``standard_normal`` is not traceable by ``torch.compile``.
    """

    GRAPH_READY = False

    def __init__(self, seed=None):
        super().__init__()

        cfg = get_cfg()
        data_cfg = get_data_cfg()

        self.seq_len = data_cfg.padded_length_in_nsamples
        self.device = cfg.device
        self.n_detectors = len(cfg.detectors)
        self.batch_size = cfg.batch_size

        self.rng = np.random.default_rng(seed)

        self.noise_target = torch.zeros(
            (self.batch_size, 1), dtype=cfg.dtype, device=cfg.device
        )

    def _sample_batch(self):
        """
        Draw one batch of white Gaussian noise and convert to FD complex.

        Returns
        -------
        torch.Tensor, shape ``(B, D, F)`` complex64
            rfft of unit-variance Gaussian TD noise with ``norm='forward'``.
        """
        arr = self.rng.standard_normal(
            (self.batch_size, self.n_detectors, self.seq_len)
        ).astype(np.float32)
        td = torch.from_numpy(arr).to(device=self.device)
        return torch.fft.rfft(td, dim=-1, norm="forward")

    @torch.no_grad()
    def forward(self):
        """
        Return a noise batch and zero targets.

        Returns
        -------
        noise_fd : torch.Tensor, shape ``(B, D, F)`` complex
        noise_target : torch.Tensor, shape ``(B, 1)`` float — all zeros
        """
        return self._sample_batch(), self.noise_target
