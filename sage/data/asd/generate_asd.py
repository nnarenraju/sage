#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Normalising-flow generative model for detector ASDs in spline
coefficient space.

Filename        : generate_asd.py
Description     : Normalising-flow generative model for detector ASDs.

Created on 2025-12-16 18:08:32

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from nflows.flows import Flow
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform
from nflows.distributions import StandardNormal
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ASDGenerator:
    """
    Normalising-flow generative model of ASD spline coefficients.

    A masked autoregressive flow (``nflows``) is fitted to spline
    coefficients of measured ASDs; samples from the flow are mapped back
    to ASDs by :meth:`spline_to_asd`.

    Parameters
    ----------
    spline_coeffs : numpy.ndarray, shape ``(N_asd, num_spline_coeffs)``
        Spline coefficients of the training ASDs.
    num_layers : int
        Number of autoregressive transform layers.
    hidden_features : int
        Hidden-layer width of each autoregressive transform.
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
        """Build the flow over the spline coefficients."""
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
        """Train the flow by maximum likelihood with Adam."""
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

    def spline_to_asd(
        self,
        coeffs: np.ndarray,
        freqs: np.ndarray,
        smoothing_spline=None,
    ) -> np.ndarray:
        """
        Convert spline coefficients to ASD values.

        The log ASD is the spline evaluated at ``len(freqs)`` points
        evenly spaced over the coefficient index range.

        Parameters
        ----------
        coeffs : numpy.ndarray
            Shape ``(num_spline_coeffs,)`` or
            ``(N_samples, num_spline_coeffs)``.
        freqs : numpy.ndarray
            Frequencies of the ASD; only their number is used.
        smoothing_spline : callable or None
            Spline evaluated instead of an interpolating spline through
            ``coeffs``.

        Returns
        -------
        numpy.ndarray, shape ``(N_samples, len(freqs))``
            ASDs; ``N_samples`` is 1 for a single coefficient vector.
        """
        from scipy.interpolate import UnivariateSpline

        coeffs = np.atleast_2d(coeffs)
        asds = []
        for c in coeffs:
            if smoothing_spline is None:
                spline = UnivariateSpline(range(len(c)), c, s=0)
            else:
                spline = smoothing_spline
            logp = spline(np.linspace(0, len(c) - 1, len(freqs)))
            asds.append(np.exp(logp))
        return np.array(asds)

    def sample_asds(self, nsamples: int, freqs: np.ndarray) -> np.ndarray:
        """
        Sample new ASDs from the trained flow.

        Parameters
        ----------
        nsamples : int
            Number of ASDs.
        freqs : numpy.ndarray
            Frequencies of the ASD (see :meth:`spline_to_asd`).

        Returns
        -------
        numpy.ndarray, shape ``(nsamples, len(freqs))``
            Sampled ASDs.
        """
        self.flow.eval()
        with torch.no_grad():
            z = self.base_dist.sample((nsamples,))
            new_coeffs = self.flow.inverse(z).cpu().numpy()
        asds = self.spline_to_asd(new_coeffs, freqs)
        return asds
