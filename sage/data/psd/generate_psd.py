#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : psds.py
Description     : Short description of the file

Created on 2025-12-16 18:08:32

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from nflows.flows import Flow
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform
from nflows.distributions import StandardNormal
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PSDGenerator:
    """
    Normalising-flow-based PSD generative model trained in log-spline space.

    Fits a masked autoregressive flow (MAF, implemented via ``nflows``) to the
    spline coefficients of real measured PSDs.  Once trained, the flow can
    generate novel plausible PSD realisations by sampling from the latent
    Gaussian and inverting through the learned transform.

    The overall pipeline is:

    1. Represent each real PSD as a set of log-spline coefficients (using
       :class:`~sage.data.psd.smoothing.LogSplineSmoothing`).
    2. Fit the MAF on those coefficients (:meth:`build_flow`,
       :meth:`train_flow`).
    3. Sample new coefficient vectors from the flow and reconstruct PSDs via
       :meth:`spline_to_psd`.

    Parameters
    ----------
    spline_coeffs : numpy.ndarray, shape ``(N_psd, num_spline_coeffs)``
        Matrix of spline coefficients from the training PSDs.
    num_layers : int
        Number of autoregressive transform layers (default ``5``).
    hidden_features : int
        Hidden-layer width in each MADE block (default ``64``).
    """

    def __init__(
        self,
        spline_coeffs: np.ndarray,
        num_layers=5,
        hidden_features=64,
    ):
        self.spline_coeffs = torch.tensor(spline_coeffs, dtype=torch.float32)
        self.num_layers = num_layers
        self.hidden_features = hidden_features
        self.flow = None
        self.learning_rate = 1e-3
        self.batch_size = 128
        self.n_epochs = 50

        self.num_coeffs = self.spline_coeffs.shape[1]

    def build_flow(self):
        """Build normalising flow in spline coefficient space."""
        transforms = []
        for _ in range(self.num_layers):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=self.num_coeffs, hidden_features=self.hidden_features
                )
            )
        transform = CompositeTransform(transforms)
        base_dist = StandardNormal([self.num_coeffs])
        self.flow = Flow(transform, base_dist)
        self.base_dist = base_dist
        logger.info(
            f"Flow built with {self.num_layers} layers and {self.num_coeffs} coefficients"
        )

    def train_flow(self):
        """Train the flow on spline coefficients."""
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.learning_rate)
        dataset = TensorDataset(self.spline_coeffs)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for (batch,) in dataloader:
                optimizer.zero_grad()
                loss = -self.flow.log_prob(batch).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            logger.info(
                f"Epoch {epoch+1}/{self.n_epochs}, loss: {total_loss / len(dataset):.6f}"
            )

    def spline_to_psd(
        self,
        coeffs: np.ndarray,
        freqs: np.ndarray,
        smoothing_spline=None,
    ) -> np.ndarray:
        """
        Convert spline coefficients to PSD values.

        Args:
            coeffs: (num_spline_coeffs,) or (N_samples, num_spline_coeffs)
            freqs: frequency array corresponding to PSD
            smoothing_spline: optional UnivariateSpline object to reconstruct PSD

        Returns:
            psd: (len(freqs),) or (N_samples, len(freqs))
        """
        from scipy.interpolate import UnivariateSpline

        coeffs = np.atleast_2d(coeffs)
        psds = []
        for c in coeffs:
            if smoothing_spline is None:
                # default: reconstruct spline from coefficients
                spline = UnivariateSpline(range(len(c)), c, s=0)
            else:
                spline = smoothing_spline
            logp = spline(np.linspace(0, len(c) - 1, len(freqs)))
            psds.append(np.exp(logp))
        return np.array(psds)

    def sample_psds(self, nsamples: int, freqs: np.ndarray) -> np.ndarray:
        """Sample new PSDs from the trained flow."""
        self.flow.eval()
        with torch.no_grad():
            z = self.base_dist.sample((nsamples,))
            new_coeffs = self.flow.inverse(z).cpu().numpy()
        psds = self.spline_to_psd(new_coeffs, freqs)
        return psds
