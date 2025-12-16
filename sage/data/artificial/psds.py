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


# Packages
import torch
import numpy as np

# Principal component analysis
from sklearn.decomposition import PCA

# Normalising flow
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform

# LOCAL
from sage.core.logger import get_logger

logger = get_logger(__name__)


class PSDGenerator:

    def __init__(self, psds, n_components=20, num_layers=5, hidden_features=64):
        # PSD realisations for training
        self.psds = psds
        # Principal component analysis
        self.pca = None
        self.n_components = n_components
        # Normalising flow
        self.num_layers = num_layers
        self.hidden_features = hidden_features
        # Learning
        self.learning_rate = 1e-3
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    def proprocess_psds(self):
        """Preprocess PSDs: log + shape-only"""
        # Log transform
        log_psds = np.log(self.psds + 1e-12)  # avoid log(0)
        # Remove scale per PSD (median=0)
        self.psds = log_psds - np.median(log_psds, axis=1, keepdims=True)

    def get_psd_componenets(self):
        """Principal Component Analysis"""
        self.pca = PCA(n_components=self.n_components)
        pca_coeffs = self.pca.fit_transform(
            log_psds_norm
        )  # shape (N_psd, n_components)
        logger.info("PCA explained variance ratio:", pca.explained_variance_ratio_)
        # Convert to torch tensor
        self.pca_coeffs = torch.tensor(pca_coeffs, dtype=torch.float32)

    def build_flow(self):
        """Build the normalizing flow in PCA space"""
        transforms = []
        for _ in range(self.num_layers):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=self.n_components, hidden_features=self.hidden_features
                )
            )

        transform = CompositeTransform(transforms)
        base_dist = StandardNormal([self.n_components])
        self.flow = Flow(transform, base_dist)

    def train_flow(self):
        """Training loop"""
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.learning_rate)

        dataset = torch.utils.data.TensorDataset(self.pca_coeffs)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.n_epochs):
            total_loss = 0
            for (batch,) in dataloader:
                optimizer.zero_grad()
                loss = -self.flow.log_prob(batch).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            logger.info(f"Epoch {epoch+1}, loss: {total_loss / len(train_data):.4f}")

    def sample_psds(nsamples=10):
        """Sampling new PSD shapes"""
        with torch.no_grad():
            z = base_dist.sample((nsamples,))
            new_coeffs = self.flow.inverse(z)  # shape (n_samples, n_components)

        # Reconstruct PSDs
        reconstructed_log_psds = self.pca.inverse_transform(new_coeffs.numpy())
        reconstructed_psds = np.exp(reconstructed_log_psds)
        return reconstructed_psds
