#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Adaptive Hard Noise Mining for GW Detection Training.

References
----------
CMA-ME : Fontaine et al., GECCO 2020.
CMA-MEGA : Fontaine et al., NeurIPS 2021.
QDax : https://github.com/adaptive-intelligent-robotics/qdax
"""

# =============================================================================
# ALGORITHM: Adaptive Hard Noise Mining with Mutual Learning
# =============================================================================
#
# GOAL
# ----
# Accumulate tens of millions of noise windows that maximise the model's
# false-alarm ranking statistic (score >= threshold), covering as many
# distinct noise types as possible, and continuously adapting to the model's
# evolving weaknesses as training progresses.
#
# KEY INSIGHT: hard noise has two signatures —
#   WHERE  : which GPS times produce high-ranking-statistic noise
#   WHAT   : which SVD spectral/temporal features are hard for the model
# Both are learned simultaneously and fed back to each other.
#
# =============================================================================
#
# SHARED STATE — SharedHardNoiseBank
# -----------------------------------
# A persistent on-disk bank of (SVD fingerprint, GPS start time, score) for
# every hard window found so far.  Both miners read from and write to it.
# It represents ALL accumulated knowledge: WHERE hard noise has been found
# and WHAT it looks like.  Saved as a versioned .npz file alongside the
# StartTimeDataset.
#
# =============================================================================
#
# MINER 1 — CMAMEMiner  (GPS Explorer, runs every N_explore epochs)
# -----------------------------------------------------------------
# PURPOSE: discover NEW GPS regions and NEW noise types the current model
# finds hard.  Always push into unexplored territory.
#
# ALGORITHM:
#   Initialisation:
#     - Fit SVD projector from random noise (captures the noise manifold).
#     - Pre-fill the SVD archive with (1 - explore_fraction) × budget GPS
#       positions drawn from the shared bank.  These are known hard GPS times;
#       seeding the archive puts the CMA in the right neighbourhoods
#       immediately and re-evaluates whether they are STILL hard under the
#       current model.  Everything above threshold is added to the accumulator.
#     - The remaining explore_fraction × budget is pure random GPS sampling.
#       Together these give ~70% new exploration / ~30% re-validation.
#
#   Main loop (CMA-ES in GPS start-time space):
#     - CMA covariance adapts to the GPS topology of hard noise: it learns
#       which GPS neighbourhoods cluster hard windows together, how wide each
#       cluster is, whether H1 and L1 hard times are correlated.
#     - The SVD archive (G×G grid) enforces diversity: different noise types
#       land in different cells; CMA reinitialises to unexplored cells when
#       it converges in one cluster.
#     - Every above-threshold window → added to accumulator AND shared bank.
#
#   Output:
#     - Fresh StartTimeDataset (this run's hard windows).
#     - Updated shared bank (enriched with new GPS regions + SVD fingerprints).
#
# =============================================================================
#
# MINER 2 — CMAMEGAMiner  (Pattern Refiner, runs every epoch between explorer)
# -----------------------------------------------------------------------------
# PURPOSE: take everything in the bank, check if it is STILL hard for the
# current model, make stale patterns harder using the model gradient, then
# efficiently find millions more windows matching known hard patterns.
#
# ALGORITHM:
#   Phase 0 — SVD projector fit + pool building:
#     Random noise is sampled to fit the SVD basis (shared with CMAMEMiner).
#     The pool (start times + SVD features) is retained for nearest-neighbour
#     lookup during gradient refinement.
#
#   Phase 1 — Bank seeding:
#     All templates from the shared bank are loaded as the starting SVD filter.
#     The filter is now pre-calibrated from iteration 1 rather than warming up.
#
#   Phase 2 — Re-evaluation and gradient refinement:
#     A sample of N bank templates is re-read from the memmap and re-scored
#     with the CURRENT model (which may have improved since the template was
#     first added).
#
#     For each FRESH template (score >= threshold):
#       → Keep.  Use as SVD filter template and GPS CMA seed.
#
#     For each STALE template (score < threshold, model has learned it):
#       → The model improved!  Make the template HARDER:
#         a. Compute d(score)/d(x) at this window via torch.autograd.
#            This gradient points in the direction that would increase the
#            model score — i.e., make the noise MORE confusing for the model.
#         b. Project gradient into SVD space:
#               grad_svd = SVD_basis @ gradient_flat    [shape: K]
#            This gives the SVD modes that most increase the score.
#         c. Step toward harder noise:
#               svd_target = svd_old + lr * normalise(grad_svd)
#         d. Find the nearest REAL noise window in the pool to svd_target
#            (nearest-neighbour in K-dim SVD space — fast matrix op).
#         e. Re-score the real noise.
#            If new_score >= threshold: replace stale template with this
#              harder refined version in the bank.
#            Else: the model has truly mastered this noise type; discard.
#
#     This is the "pathological refinement" loop: templates are continuously
#     pushed to the hardest real noise the current model struggles with.
#
#   Phase 3 — Main mining loop:
#     Each iteration is EXPLORE or EXPLOIT.
#
#     EXPLOIT (exploit_fraction of iterations):
#       CMA-ES in GPS start-time space, initialised from bank GPS positions.
#       Same mechanism as CMAMEMiner but seeded from the bank rather than
#       from random starts.  Learns the local GPS covariance of hard-noise
#       clusters and drills deeper into them.
#       Reinitialises to a new bank GPS seed when the cluster is exhausted.
#       → All above-threshold windows added to bank + accumulator.
#
#     EXPLORE (1 - exploit_fraction of iterations, two sub-modes):
#       Sub-mode A — SVD-filtered random scan (1 - pure_explore_fraction):
#         Random GPS sampling → SVD project → keep only windows near the
#         bank's SVD patterns → model score the passing subset.
#         Finds more of the SAME types of hard noise as the bank, very
#         efficiently (few wasted model calls).
#
#       Sub-mode B — Pure random scan (pure_explore_fraction):
#         Random GPS sampling with NO SVD filter.
#         Deliberately finds novel hard noise types not yet in the bank.
#         Yield per model call is lower but these windows seed future bank
#         entries of entirely new noise patterns.
#
#     Both explore sub-modes → above-threshold windows added to bank + accumulator.
#
# =============================================================================
#
# TRAINING SCHEDULE (recommended)
# ---------------------------------
#  Epoch 10   : CMAMEMiner (explore_fraction=1.0, bank empty → pure exploration)
#  Epoch 11-20: CMAMEGAMiner every epoch (refines + expands; bank growing fast)
#  Epoch 20   : CMAMEMiner (explore_fraction=0.7, 30% seeds from bank)
#  Epoch 21-30: CMAMEGAMiner every epoch
#  ...
#  Epoch 80+  : CMAMEGAMiner only (final hardening: 100% pattern-matched mining)
#
# CROSS-RUN ACCUMULATION
# -----------------------
# Each mining run produces a fresh StartTimeDataset that replaces the previous
# one in the noise sampler (set_hard_dataset with epoch-versioned save_dir).
# The SharedHardNoiseBank persists SEPARATELY and grows across all runs —
# never reset, only pruned by score when over capacity.
#
# REACHING 40M+ SAMPLES
# ----------------------
# The bank pre-filter means later runs spend almost no model time on easy
# noise.  Combined with exploit (GPS CMA drilling) and gradient refinement
# (templates always pushed to the model's current hardest), yield per model
# call increases with each run.  Set target_samples in CMAMEGAMiner and let
# it run until satisfied; reduce threshold early (2.0-3.0) to fill the bank
# quickly, raise later (5.0+) to concentrate on the extreme tail.
# =============================================================================

import math
import json

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from pycbc import DYN_RANGE_FAC

from sage.core.config import get_cfg
from sage.core.pipeline import GWBatch, Grid, ProcessingState
from sage.data.noise.lowfar_noise import (
    StartTimeDataset,
    _MiningReader,
)


# ---------------------------------------------------------------------------
# make_miner_preprocessor
# ---------------------------------------------------------------------------

def make_miner_preprocessor(processor, signal_sampler=None):
    """
    Build a preprocessing callable that exactly mirrors the training loop's
    noise pipeline (whitening → IFFT → multirate *or* coarse FD selection →
    whitening, depending on the signal sampler's configuration).

    This ensures the miner operates on noise shaped identically to what the
    network sees during training.  Pass the result as the ``processor``
    argument to any miner's ``mine()`` call.

    Parameters
    ----------
    processor : nn.Module
        The same Preprocessor used in training (FiducialWhitening + optional
        MultirateSampler / MultibandSelector chain).
    signal_sampler : nn.Module or None
        Signal sampler from the training run.  When provided, ``output_state``
        and ``selector`` are read from it to detect worst-case multibanding
        (FD_COARSE) pipelines and apply the coarse-grid selector to the noise
        before whitening — exactly as SageVanillaTraining does.

    Returns
    -------
    callable
        ``preprocess_fn(noise_fd) -> net_input``

        * ``noise_fd``   : ``(B, D, F)`` complex64 FD tensor (full resolution)
        * ``net_input``  : real float32 tensor ready for the model
                           — ``(B, D, T)`` for TD pipelines
                           — ``(B, 2·D, N_coarse)`` for FD_COARSE pipelines
    """
    if signal_sampler is not None:
        initial_state = getattr(
            signal_sampler, "output_state", ProcessingState(Grid.FD_UNIFORM)
        )
        selector      = getattr(signal_sampler, "selector", None)
        freqs         = selector.coarse_freqs    if selector is not None else None
        coarse_idx    = selector.coarse_indices  if selector is not None else None
    else:
        initial_state = ProcessingState(Grid.FD_UNIFORM)
        selector      = None
        freqs         = None
        coarse_idx    = None

    def preprocess_fn(noise_fd: torch.Tensor) -> torch.Tensor:
        if selector is not None:
            noise_fd = selector(noise_fd)
        batch = GWBatch(
            noise_fd,
            state          = initial_state,
            freqs          = freqs,
            coarse_indices = coarse_idx,
        )
        batch = processor(batch)
        return batch.to_network_input()

    return preprocess_fn


# ---------------------------------------------------------------------------
# NoiseSVDProjector
# ---------------------------------------------------------------------------

class NoiseSVDProjector:
    """
    Projects preprocessed noise into a low-dimensional behavioral descriptor
    space via truncated SVD followed by PCA.

    The projector must be fitted on an initial pool of noise samples before
    it can transform new ones.  The 2-D output is used to index the QD archive.

    Parameters
    ----------
    n_components : int
        Number of SVD components to retain (intermediate representation).
    pca_dims : int
        Final descriptor dimensionality (should be 2 for a 2-D archive grid).
    """

    def __init__(self, n_components: int = 32, pca_dims: int = 2):
        self.n_components = n_components
        self.pca_dims = pca_dims

        # Set after fit()
        self.svd_components: np.ndarray | None = None   # (n_components, D*T)
        self.pca_mean:        np.ndarray | None = None   # (n_components,)
        self.pca_components:  np.ndarray | None = None   # (pca_dims, n_components)
        self.is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, noise_pool: torch.Tensor) -> np.ndarray:
        """
        Fit SVD + PCA on a pool of preprocessed noise samples.

        Parameters
        ----------
        noise_pool : torch.Tensor, shape (N, D, T) or (N, C, F)
            Preprocessed network-input tensors (float32, CPU or GPU).

        Returns
        -------
        descriptors : np.ndarray, shape (N, pca_dims)
            2-D descriptors for all pool samples.
        """
        from sklearn.decomposition import TruncatedSVD, PCA

        N = noise_pool.shape[0]
        X = noise_pool.reshape(N, -1).float().cpu().numpy()

        k = min(self.n_components, min(X.shape) - 1)
        svd = TruncatedSVD(n_components=k, random_state=0)
        features = svd.fit_transform(X)          # (N, k)
        self.svd_components = svd.components_    # (k, D*T)

        k_pca = min(self.pca_dims, k)
        pca = PCA(n_components=k_pca, random_state=0)
        descriptors = pca.fit_transform(features)  # (N, pca_dims)
        self.pca_mean       = pca.mean_
        self.pca_components = pca.components_
        self.is_fitted = True
        return descriptors

    # ------------------------------------------------------------------
    def transform(self, noise: torch.Tensor) -> np.ndarray:
        """
        Project noise windows to the 2-D descriptor space.

        Parameters
        ----------
        noise : torch.Tensor, shape (B, D, T) or (B, C, F)

        Returns
        -------
        np.ndarray, shape (B, pca_dims)
        """
        if not self.is_fitted:
            raise RuntimeError("NoiseSVDProjector.fit() must be called first.")
        B = noise.shape[0]
        X = noise.reshape(B, -1).float().cpu().numpy()
        features = X @ self.svd_components.T                         # (B, k)
        descriptors = (features - self.pca_mean) @ self.pca_components.T  # (B, pca_dims)
        return descriptors

    # ------------------------------------------------------------------
    def svd_encode(self, noise: torch.Tensor) -> np.ndarray:
        """
        Return the K SVD coefficients (before PCA) for use in CMA-MEGA.

        Parameters
        ----------
        noise : torch.Tensor, shape (B, D, T)

        Returns
        -------
        np.ndarray, shape (B, n_components)
        """
        if not self.is_fitted:
            raise RuntimeError("NoiseSVDProjector.fit() must be called first.")
        B = noise.shape[0]
        X = noise.reshape(B, -1).float().cpu().numpy()
        return X @ self.svd_components.T

    # ------------------------------------------------------------------
    def svd_decode(self, coeffs: np.ndarray, original_shape) -> torch.Tensor:
        """
        Reconstruct the approximate noise from SVD coefficients.

        Parameters
        ----------
        coeffs : np.ndarray, shape (B, n_components)
        original_shape : tuple — target shape (B, D, T)

        Returns
        -------
        torch.Tensor, shape (B, D, T) float32
        """
        X_approx = coeffs @ self.svd_components        # (B, D*T)
        B = original_shape[0]
        rest = math.prod(original_shape[1:])
        return torch.from_numpy(X_approx.reshape(B, *original_shape[1:]).astype(np.float32))


# ---------------------------------------------------------------------------
# _GenotypeMapper
# ---------------------------------------------------------------------------

class _GenotypeMapper:
    """
    Bidirectional mapping between a continuous genotype in R^D and discrete
    (start_indices, segment_ids) pairs consumed by _MiningReader.

    Each component g_d ∈ R maps to a fractional position p_d = sigmoid(g_d)
    in [0, 1], which is then linearly interpolated over all valid sample
    positions in detector d's memmap (accounting for segment boundaries).

    Parameters
    ----------
    seg_index : list of structured np.ndarray
        One element per detector; each array has fields
        ['idx', 'start', 'end', 'nsamples'].
    seq_len : int
        Number of samples per noise window.
    """

    def __init__(self, seg_index, seq_len: int):
        self.seq_len = seq_len
        self.D = len(seg_index)

        self.cum_valid:     list[np.ndarray] = []   # (n_segs+1,) cumulative
        self.abs_starts:    list[np.ndarray] = []   # (n_segs,)  memmap starts
        self.seg_ids:       list[np.ndarray] = []   # (n_segs,)  segment IDs
        self.valid_per_seg: list[np.ndarray] = []   # (n_segs,)  valid samples
        self.N:             list[int]        = []   # total valid per detector

        for seg_arr in seg_index:
            valid = np.maximum(0, seg_arr["nsamples"].astype(np.int64) - seq_len)
            cum   = np.concatenate([[0], np.cumsum(valid)])
            self.cum_valid.append(cum)
            self.abs_starts.append(seg_arr["start"].astype(np.int64))
            self.seg_ids.append(seg_arr["idx"].astype(np.int64))
            self.valid_per_seg.append(valid)
            self.N.append(int(cum[-1]))

    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return np.log(p / (1.0 - p))

    # ------------------------------------------------------------------
    def decode(self, genotypes: np.ndarray):
        """
        Map (B, D) genotype array to start indices and segment ids.

        Returns
        -------
        starts : (B, D) int64
        segs   : (B, D) int64
        """
        B, D = genotypes.shape
        starts = np.zeros((B, D), dtype=np.int64)
        segs   = np.zeros((B, D), dtype=np.int64)

        for d in range(D):
            p = self._sigmoid(genotypes[:, d])
            linear_pos = (p * self.N[d]).astype(np.int64)
            linear_pos = np.clip(linear_pos, 0, max(self.N[d] - 1, 0))

            cum      = self.cum_valid[d]
            arr_idx  = np.searchsorted(cum, linear_pos, side="right") - 1
            arr_idx  = np.clip(arr_idx, 0, len(self.abs_starts[d]) - 1)

            offset   = linear_pos - cum[arr_idx]
            max_off  = np.maximum(0, self.valid_per_seg[d][arr_idx] - 1)
            offset   = np.clip(offset, 0, max_off)

            starts[:, d] = self.abs_starts[d][arr_idx] + offset
            segs[:, d]   = self.seg_ids[d][arr_idx]

        return starts, segs

    # ------------------------------------------------------------------
    def encode(self, starts: np.ndarray, segs: np.ndarray) -> np.ndarray:
        """
        Map (B, D) start indices to (B, D) continuous genotype.

        Parameters
        ----------
        starts : (B, D) int64
        segs   : (B, D) int64  — segment IDs (from seg_arr['idx'])

        Returns
        -------
        genotypes : (B, D) float64
        """
        B, D = starts.shape
        genotypes = np.zeros((B, D), dtype=np.float64)

        for d in range(D):
            # Build segment-ID → array-index lookup
            id_to_pos = {int(sid): i for i, sid in enumerate(self.seg_ids[d])}
            arr_idx = np.array(
                [id_to_pos.get(int(s), 0) for s in segs[:, d]], dtype=np.int64
            )
            offset      = starts[:, d] - self.abs_starts[d][arr_idx]
            linear_pos  = self.cum_valid[d][arr_idx] + offset
            p           = linear_pos.astype(np.float64) / max(self.N[d], 1)
            genotypes[:, d] = self._logit(p)

        return genotypes


# ---------------------------------------------------------------------------
# _CMAES   (pure NumPy, ported from QDax baselines/cmaes.py)
# ---------------------------------------------------------------------------

class _CMAESState:
    """Mutable CMA-ES state (mean, covariance, sigma, evolution paths)."""

    __slots__ = [
        "mean", "cov", "sigma", "p_c", "p_s",
        "invsqrt_cov", "eigenvalues", "num_updates",
    ]

    def __init__(self, mean, cov, sigma, p_c, p_s, num_updates=0):
        self.mean        = np.asarray(mean, dtype=np.float64).copy()
        self.cov         = np.asarray(cov,  dtype=np.float64).copy()
        self.sigma       = float(sigma)
        self.p_c         = np.asarray(p_c, dtype=np.float64).copy()
        self.p_s         = np.asarray(p_s, dtype=np.float64).copy()
        self.num_updates = int(num_updates)
        self._refresh_invsqrt()

    def _refresh_invsqrt(self):
        eigvals, eigvecs = np.linalg.eigh(self.cov)
        eigvals = np.maximum(eigvals, 1e-12)
        self.invsqrt_cov = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        self.eigenvalues = eigvals


class _CMAES:
    """
    Pure-NumPy CMA-ES (Hansen, 2016).  Ported from QDax without JAX.

    Parameters
    ----------
    search_dim : int
        Dimension of the search space.
    population_size : int
        Number of offspring per generation.
    init_sigma : float
        Initial step size.
    bias_weights : bool
        Use log-rank weights (recommended).
    delay_eigen_decomp : bool
        Recompute invsqrt_cov only every `_eigen_period` updates.
    """

    def __init__(
        self,
        search_dim: int,
        population_size: int,
        init_sigma: float = 0.5,
        bias_weights: bool = True,
        delay_eigen_decomp: bool = True,
    ):
        self.n   = search_dim
        self.lam = population_size
        self.mu  = population_size // 2
        self.init_sigma = init_sigma

        # Weights (log-rank)
        if bias_weights:
            raw = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        else:
            raw = np.ones(self.mu)
        self.weights  = raw / raw.sum()
        self.mu_eff   = 1.0 / (self.weights ** 2).sum()

        # CMA adaptation parameters
        n = self.n
        self.c_s  = (self.mu_eff + 2) / (n + self.mu_eff + 5)
        self.d_s  = (
            1
            + 2 * max(0.0, float(np.sqrt((self.mu_eff - 1) / (n + 1))) - 1.0)
            + self.c_s
        )
        self.c_c  = (4 + self.mu_eff / n) / (n + 4 + 2 * self.mu_eff / n)
        self.c_1  = 2 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((n + 2) ** 2 + self.mu_eff),
        )
        self.chi_n = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        if delay_eigen_decomp:
            self._eigen_period = max(
                1, int(0.5 * self.lam / (n * (self.c_1 + self.c_mu)))
            )
        else:
            self._eigen_period = 1

    # ------------------------------------------------------------------
    def init(self, mean: np.ndarray | None = None) -> _CMAESState:
        if mean is None:
            mean = np.zeros(self.n, dtype=np.float64)
        return _CMAESState(
            mean=mean, cov=np.eye(self.n),
            sigma=self.init_sigma,
            p_c=np.zeros(self.n), p_s=np.zeros(self.n),
        )

    # ------------------------------------------------------------------
    def sample(self, state: _CMAESState, rng: np.random.Generator) -> np.ndarray:
        """Draw lam samples from N(mean, sigma² · C). Returns (lam, n)."""
        z = rng.standard_normal((self.lam, self.n))
        L = np.linalg.cholesky(state.cov + 1e-10 * np.eye(self.n))
        return state.mean + state.sigma * (z @ L.T)

    # ------------------------------------------------------------------
    def update(
        self,
        state:           _CMAESState,
        sorted_samples:  np.ndarray,          # (lam, n), best first
        mask:            np.ndarray | None = None,  # (mu,) float, default ones
    ) -> _CMAESState:
        """Return an updated CMA-ES state."""
        mu = self.mu
        old_mean = state.mean
        w = self.weights if mask is None else self.weights * mask[:mu]
        w_sum = w.sum()
        if w_sum < 1e-12:
            return state
        w = w / w_sum

        new_mean = (w[:, None] * sorted_samples[:mu]).sum(0)
        step     = (new_mean - old_mean) / state.sigma

        # Evolution path for sigma
        p_s = (
            (1 - self.c_s) * state.p_s
            + np.sqrt(self.c_s * (2 - self.c_s) * self.mu_eff)
            * (state.invsqrt_cov @ step)
        )

        norm_ps = float(np.linalg.norm(p_s))
        denom   = np.sqrt(1 - (1 - self.c_s) ** (2 * (state.num_updates + 1)))
        h_sigma = float(norm_ps / (denom + 1e-14) < (1.4 + 2 / (self.n + 1)) * self.chi_n)

        # Evolution path for covariance
        p_c = (1 - self.c_c) * state.p_c + h_sigma * np.sqrt(
            self.c_c * (2 - self.c_c) * self.mu_eff
        ) * step

        # Covariance matrix update
        y   = (sorted_samples[:mu] - old_mean) / state.sigma
        cov = (
            (1 - self.c_1 - self.c_mu) * state.cov
            + self.c_1 * np.outer(p_c, p_c)
            + self.c_mu * (w[:, None] * y).T @ y
        )

        # Step-size adaptation
        sigma = float(
            state.sigma
            * np.exp(self.c_s / self.d_s * (norm_ps / self.chi_n - 1))
        )
        sigma = max(sigma, 1e-14)

        new_state = _CMAESState(
            mean=new_mean, cov=cov, sigma=sigma, p_c=p_c, p_s=p_s,
            num_updates=state.num_updates + 1,
        )
        return new_state

    # ------------------------------------------------------------------
    def stop_condition(self, state: _CMAESState) -> bool:
        """True when CMA-ES has converged or degenerated."""
        return (
            state.sigma < 1e-12
            or bool(np.any(np.isnan(state.mean)))
            or state.num_updates > 500 * self.n
        )


# ---------------------------------------------------------------------------
# _QDArchive
# ---------------------------------------------------------------------------

class _QDArchive:
    """
    MAP-Elites archive with a regular G×G grid in 2-D SVD-descriptor space.

    Each cell holds the single highest-scoring noise window found in that
    region of behavioral space.

    Parameters
    ----------
    n_cells_per_dim : int  (G)
    n_detectors : int      (D)
    """

    def __init__(self, n_cells_per_dim: int = 32, n_detectors: int = 2):
        G = n_cells_per_dim
        self.G       = G
        self.n_total = G * G
        self.D       = n_detectors

        self.fitnesses  = np.full(self.n_total, -np.inf, dtype=np.float32)
        self.genotypes  = np.zeros((self.n_total, n_detectors), dtype=np.float64)
        self.starts     = np.zeros((self.n_total, n_detectors), dtype=np.int64)
        self.segs       = np.zeros((self.n_total, n_detectors), dtype=np.int64)
        self.desc_lo:   np.ndarray | None = None
        self.desc_hi:   np.ndarray | None = None

    # ------------------------------------------------------------------
    def _init_bounds(self, descriptors: np.ndarray):
        span = np.abs(descriptors.max(0) - descriptors.min(0))
        self.desc_lo = descriptors.min(0) - 0.5 * span
        self.desc_hi = descriptors.max(0) + 0.5 * span

    def _desc_to_cell(self, descriptors: np.ndarray) -> np.ndarray:
        """(B, 2) descriptors → (B,) integer cell ids."""
        span = self.desc_hi - self.desc_lo
        span = np.where(span < 1e-8, 1.0, span)
        norm = np.clip((descriptors - self.desc_lo) / span, 0.0, 1.0 - 1e-7)
        i0   = (norm[:, 0] * self.G).astype(np.int32)
        i1   = (norm[:, 1] * self.G).astype(np.int32)
        return i0 * self.G + i1

    # ------------------------------------------------------------------
    def update(
        self,
        genotypes:   np.ndarray,   # (B, D)
        starts:      np.ndarray,   # (B, D)
        segs:        np.ndarray,   # (B, D)
        fitnesses:   np.ndarray,   # (B,)
        descriptors: np.ndarray,   # (B, 2)
    ):
        """
        Attempt to insert each candidate into its archive cell.

        Returns
        -------
        n_improved : int
        improvements : (B,) float32  — +inf for new cells
        """
        if self.desc_lo is None:
            self._init_bounds(descriptors)

        cell_ids     = self._desc_to_cell(descriptors)
        improvements = np.zeros(len(fitnesses), dtype=np.float32)
        n_improved   = 0

        for i in range(len(fitnesses)):
            cid = int(cell_ids[i])
            old = float(self.fitnesses[cid])
            if not np.isfinite(old):
                improvements[i] = np.inf          # new cell
            else:
                improvements[i] = fitnesses[i] - old

            if fitnesses[i] > old:
                self.fitnesses[cid]  = fitnesses[i]
                self.genotypes[cid]  = genotypes[i]
                self.starts[cid]     = starts[i]
                self.segs[cid]       = segs[i]
                n_improved += 1

        return n_improved, improvements

    # ------------------------------------------------------------------
    def sample_elite(self, rng: np.random.Generator) -> np.ndarray | None:
        """Return the genotype of a uniformly-random filled cell (or None)."""
        filled = np.where(np.isfinite(self.fitnesses))[0]
        if len(filled) == 0:
            return None
        return self.genotypes[rng.integers(0, len(filled))].copy()

    @property
    def n_filled(self) -> int:
        return int(np.sum(np.isfinite(self.fitnesses)))



# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _improvement_ranking(scores: np.ndarray, improvements: np.ndarray) -> np.ndarray:
    """
    CMA-ME improvement criterion: rank by archive improvement, but put new
    cells (improvement == +inf) first with an additional offset.

    Returns a ranking array (higher = better).
    """
    new_cell = np.isinf(improvements)
    ranking  = np.where(new_cell, scores, improvements)
    if new_cell.any():
        offset  = float(ranking[np.isfinite(ranking)].max()
                        - ranking[np.isfinite(ranking)].min()) if np.isfinite(ranking).any() else 1.0
        ranking = np.where(new_cell, ranking + offset, ranking)
    return ranking

# ---------------------------------------------------------------------------
# _StreamingAccumulator  (shared by both miners)
# ---------------------------------------------------------------------------

class _StreamingAccumulator:
    """
    Accumulates (starts, segs, scores) for every evaluated noise window
    that exceeds a score threshold.  Prunes to top-``max_samples`` every
    ``prune_every`` additions to keep memory bounded.

    This replaces the old "only save archive cells" approach so that ALL
    above-threshold windows found during mining are returned.
    """

    def __init__(self, max_samples: int = 5_000_000, prune_every: int = 10_000):
        self.max_samples = max_samples
        self.prune_every = prune_every
        self._starts:  list[np.ndarray] = []
        self._segs:    list[np.ndarray] = []
        self._scores:  list[np.ndarray] = []
        self._n_added  = 0
        self._total    = 0

    def add(self, starts: np.ndarray, segs: np.ndarray, scores: np.ndarray, threshold: float):
        """Add any windows above threshold from a batch."""
        mask = scores >= threshold
        if not mask.any():
            return
        self._starts.append(starts[mask])
        self._segs.append(segs[mask])
        self._scores.append(scores[mask].astype(np.float32))
        self._n_added += int(mask.sum())
        self._total   += int(mask.sum())

        if self._n_added >= self.prune_every:
            self._prune()

    def _prune(self):
        if not self._starts:
            return
        all_sc = np.concatenate(self._scores)
        if len(all_sc) > self.max_samples:
            k       = self.max_samples
            top_idx = np.argpartition(all_sc, -k)[-k:]
            all_st  = np.concatenate(self._starts, axis=0)
            all_sg  = np.concatenate(self._segs,   axis=0)
            self._starts  = [all_st[top_idx]]
            self._segs    = [all_sg[top_idx]]
            self._scores  = [all_sc[top_idx]]
        self._n_added = 0

    def to_dataset(self, noise_sampler, reader) -> StartTimeDataset:
        """Build a StartTimeDataset from all accumulated windows."""
        cfg = get_cfg()
        if not self._starts:
            return reader._empty_dataset(noise_sampler)
        all_starts = np.concatenate(self._starts, axis=0)
        all_segs   = np.concatenate(self._segs,   axis=0)
        all_scores = np.concatenate(self._scores)
        gps        = reader.gps_from_starts(all_starts, all_segs)
        sort_idx   = np.argsort(all_scores)[::-1]
        return StartTimeDataset(
            detectors       = cfg.detectors,
            start_indices   = all_starts[sort_idx],
            segment_indices = all_segs[sort_idx],
            gps_times       = gps[sort_idx],
            scores          = all_scores[sort_idx],
            bin_files       = [str(p) for p in noise_sampler.bin_files],
            sample_rate     = reader.sample_rate,
            seq_len         = noise_sampler.seq_len,
        )

    def __len__(self):
        return sum(len(s) for s in self._scores)


# ---------------------------------------------------------------------------
# CMAMEMiner
# ---------------------------------------------------------------------------

class CMAMEMiner:
    """
    CMA-ME (Covariance Matrix Adaptation MAP-Elites) hard noise miner.

    Searches in a continuous start-time space (genotype ∈ R^D, one dimension
    per detector) for noise windows that maximise the frozen model's ranking
    statistic while maintaining diversity across noise-feature space.

    Algorithm
    ---------
    The behavioral descriptor is the 2-D SVD + PCA projection of the
    preprocessed noise (whitened + multirate compressed).  A G × G
    MAP-Elites archive over this 2-D space drives diversity.  CMA-ES adapts
    its covariance toward start-time positions that improve the archive —
    discovering harder and harder windows in each region of noise-feature
    space.

    *All* windows scoring above ``threshold`` are collected into the output
    ``StartTimeDataset``, not just the best per archive cell.  The archive
    is the internal diversity engine; the accumulator is what you get back.

    Parameters
    ----------
    n_svd_components : int
        SVD rank for the behavioral descriptor computation.
    n_init_batches : int
        Random batches used to fit the SVD projector and seed the archive
        before the CMA loop begins.  More batches = better SVD coverage.
    n_iterations : int
        CMA update steps.  Total windows evaluated ≈
        ``(n_init_batches + n_iterations) × batch_size``.
    batch_size : int
        Offspring evaluated per CMA step.
    sigma_g : float
        Initial CMA step size (logit-space start-time coordinates).
    min_count : int
        Minimum emissions before reinitialisation is considered.
    max_count : int or None
        Maximum emissions before forced reinitialisation.
    grid_size : int
        G in the G × G archive grid.
    threshold : float
        Minimum model score for a window to enter the output dataset.
    max_samples : int
        Hard cap on the output dataset size (streaming top-K prune).
    autocast : bool
        Float16 AMP during inference.
    """

    def __init__(
        self,
        n_svd_components: int   = 32,
        n_init_batches:   int   = 200,
        n_iterations:     int   = 2000,
        batch_size:       int   = 36,
        sigma_g:          float = 0.5,
        min_count:        int   = 1,
        max_count:        int | None = None,
        grid_size:        int   = 32,
        threshold:        float = 3.0,
        max_samples:      int   = 5_000_000,
        explore_fraction: float = 0.7,
        autocast:         bool  = True,
    ):
        self.n_svd_components = n_svd_components
        self.n_init_batches   = n_init_batches
        self.n_iterations     = n_iterations
        self.batch_size       = batch_size
        self.sigma_g          = sigma_g
        self.min_count        = min_count
        self.max_count        = max_count if max_count is not None else float("inf")
        self.grid_size        = grid_size
        self.threshold        = threshold
        self.max_samples      = max_samples
        self.explore_fraction = explore_fraction
        self.autocast         = autocast

    # ------------------------------------------------------------------
    @torch.no_grad()
    def mine(
        self,
        model,
        noise_sampler,
        processor,
        device: str,
        signal_sampler=None,
        shared_bank: SharedHardNoiseBank | None = None,
    ) -> StartTimeDataset:
        """
        Run a CMA-ME mining pass and return all above-threshold windows.

        Parameters
        ----------
        model : nn.Module
            Frozen Sage model (put into eval internally).
        noise_sampler : MemmapNoiseSampler
        processor : nn.Module
            Same Preprocessor used in training.
        device : str
        signal_sampler : nn.Module or None
            Signal sampler from training.  Always pass this.
        shared_bank : SharedHardNoiseBank or None
            Shared knowledge bank.  When provided:
            - (1 - explore_fraction) of the init budget re-evaluates bank GPS
              positions with the current model (checks if still hard).
            - explore_fraction of the init budget is pure random exploration.
            - All above-threshold findings are added to the bank.
            Pass the same bank object to CMAMEGAMiner so both miners share
            accumulated knowledge.

        Returns
        -------
        StartTimeDataset
            Every window above ``threshold`` found during this run.
        """
        reader        = _MiningReader(noise_sampler)
        was_training  = model.training
        model.eval()
        preprocess_fn = make_miner_preprocessor(processor, signal_sampler)

        cast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.autocast else nullcontext()
        )

        D       = reader.n_detectors
        rng     = np.random.default_rng()
        mapper  = _GenotypeMapper(reader.seg_index, reader.seq_len)
        cmaes   = _CMAES(search_dim=D, population_size=self.batch_size, init_sigma=self.sigma_g)
        proj    = NoiseSVDProjector(n_components=self.n_svd_components, pca_dims=2)
        archive = _QDArchive(n_cells_per_dim=self.grid_size, n_detectors=D)
        accum   = _StreamingAccumulator(max_samples=self.max_samples)

        # ── Phase 1: Seeding — random + bank GPS re-evaluation ───────────
        n_bank_batches   = 0
        n_random_batches = self.n_init_batches

        if shared_bank is not None and len(shared_bank) > 0:
            # (1 - explore_fraction) of the budget re-evaluates known hard GPS
            n_bank_samples   = max(0, int(self.n_init_batches * self.batch_size * (1 - self.explore_fraction)))
            n_bank_batches   = max(0, n_bank_samples // self.batch_size)
            n_random_batches = self.n_init_batches - n_bank_batches

        print(
            f"[CMA-ME] Init: {n_random_batches} random + "
            f"{n_bank_batches} bank-seed batches …"
        )
        pool_x, pool_starts, pool_segs, pool_scores = [], [], [], []

        # Random exploration batches
        for _ in tqdm(range(n_random_batches), desc="CMA-ME random", leave=False):
            starts, segs = reader.random_starts(self.batch_size)
            noise_fd     = reader.read_batch(starts, segs)
            x            = preprocess_fn(noise_fd)
            with cast:
                out = model(x)
            sc = out[0].squeeze(1).float().cpu().numpy()
            pool_x.append(x.float().cpu())
            pool_starts.append(starts)
            pool_segs.append(segs)
            pool_scores.append(sc)
            accum.add(starts, segs, sc, self.threshold)

        # Bank GPS re-evaluation batches
        if n_bank_batches > 0 and shared_bank is not None:
            for _ in tqdm(range(n_bank_batches), desc="CMA-ME bank seeds", leave=False):
                starts, segs = shared_bank.sample_starts(self.batch_size, rng)
                if starts is None:
                    break
                noise_fd = reader.read_batch(starts, segs)
                x        = preprocess_fn(noise_fd)
                with cast:
                    out = model(x)
                sc = out[0].squeeze(1).float().cpu().numpy()
                pool_x.append(x.float().cpu())
                pool_starts.append(starts)
                pool_segs.append(segs)
                pool_scores.append(sc)
                accum.add(starts, segs, sc, self.threshold)
                # Update bank: these windows were re-evaluated with current model
                svd_feat = proj.svd_encode(x.float().cpu()) if proj.is_fitted else None
                if svd_feat is not None and (sc >= self.threshold).any():
                    mask = sc >= self.threshold
                    shared_bank.add(svd_feat[mask], starts[mask], segs[mask], sc[mask])

        all_x      = torch.cat(pool_x, 0)
        all_starts = np.concatenate(pool_starts, 0)
        all_segs   = np.concatenate(pool_segs,   0)
        all_scores = np.concatenate(pool_scores).astype(np.float32)

        # Fit SVD projector and seed archive
        print("[CMA-ME] Fitting SVD projector …")
        descs = proj.fit(all_x)
        genos = mapper.encode(all_starts, all_segs)
        archive.update(genos, all_starts, all_segs, all_scores, descs)

        # Add all above-threshold to shared bank (now SVD projector is fitted)
        if shared_bank is not None:
            above = all_scores >= self.threshold
            if above.any():
                shared_bank.add(
                    proj.svd_encode(all_x[above]),
                    all_starts[above], all_segs[above], all_scores[above]
                )

        best_init = archive.fitnesses[np.isfinite(archive.fitnesses)]
        print(
            f"[CMA-ME] Archive seeded: {archive.n_filled}/{archive.n_total} cells | "
            f"best score = {best_init.max() if len(best_init) else float('nan'):.3f} | "
            f"above threshold so far: {len(accum):,}"
        )

        # ── Phase 2: CMA-ME main loop ────────────────────────────────────
        init_mean = archive.sample_elite(rng)
        cma_state = cmaes.init(mean=init_mean if init_mean is not None else np.zeros(D))
        emit_count = 0
        log_every  = max(1, self.n_iterations // 10)

        print(f"[CMA-ME] Main loop: {self.n_iterations} iterations …")
        for i in tqdm(range(self.n_iterations), desc="CMA-ME", leave=False):

            candidates   = cmaes.sample(cma_state, rng)
            starts, segs = mapper.decode(candidates)

            noise_fd = reader.read_batch(starts, segs)
            x        = preprocess_fn(noise_fd)
            with cast:
                out = model(x)
            scores = out[0].squeeze(1).float().cpu().numpy()

            descs = proj.transform(x.float().cpu())
            _, improvements = archive.update(
                candidates, starts, segs, scores.astype(np.float32), descs
            )

            # Accumulate all above-threshold windows + update shared bank
            accum.add(starts, segs, scores, self.threshold)
            if shared_bank is not None:
                above = scores >= self.threshold
                if above.any():
                    shared_bank.add(
                        proj.svd_encode(x[above].float().cpu()),
                        starts[above], segs[above], scores[above]
                    )

            # CMA improvement ranking and update
            ranking      = _improvement_ranking(scores, improvements)
            sorted_idx   = np.argsort(ranking)[::-1]
            cma_state    = cmaes.update(cma_state, candidates[sorted_idx])
            emit_count  += 1

            # Reinitialise on stagnation / convergence
            reinit = (
                (np.all(improvements < 0) and emit_count > self.min_count)
                or emit_count > self.max_count
                or cmaes.stop_condition(cma_state)
            )
            if reinit:
                elite     = archive.sample_elite(rng)
                cma_state = cmaes.init(mean=elite if elite is not None else np.zeros(D))
                emit_count = 0

            if (i + 1) % log_every == 0:
                best = archive.fitnesses[np.isfinite(archive.fitnesses)]
                print(
                    f"  [iter {i+1:,}/{self.n_iterations}] "
                    f"cells: {archive.n_filled}/{archive.n_total} | "
                    f"best: {best.max():.3f} | "
                    f"accumulated: {len(accum):,}"
                )

        if was_training:
            model.train()

        dataset = accum.to_dataset(noise_sampler, reader)
        print(
            f"[CMA-ME] Done — {len(dataset):,} windows above threshold "
            f"{self.threshold:.3f} | best score: {dataset.scores.max():.3f}"
        )
        return dataset


# ---------------------------------------------------------------------------
# SharedHardNoiseBank
# ---------------------------------------------------------------------------

class SharedHardNoiseBank:
    """
    Persistent shared state between CMAMEMiner and CMAMEGAMiner.

    Stores SVD fingerprints, GPS start times, and last-recorded model scores
    for every hard noise window accumulated across all mining runs.  Both
    miners read from and write to it; it is never reset, only pruned.

    Persistence
    -----------
    ``save`` / ``load`` store a compressed .npz so the bank survives restarts.
    Recommended to save alongside the StartTimeDataset after each mining run.

    Size management
    ---------------
    When the bank exceeds ``max_bank_size``, entries are pruned keeping the
    highest-scoring half plus a random half.  This retains the hardest known
    noise while maintaining diversity.

    Parameters
    ----------
    K : int
        SVD latent dimension (must match the NoiseSVDProjector used).
    max_bank_size : int
        Maximum entries stored in memory.
    max_query_size : int
        Entries sub-sampled for nearest-neighbour distance queries to keep
        the (B × N_bank × K) distance matrix under ~5 ms on CPU.
    """

    def __init__(self, K: int, max_bank_size: int = 100_000, max_query_size: int = 10_000):
        self.K              = K
        self.max_bank_size  = max_bank_size
        self.max_query_size = max_query_size
        self._svd:    list[np.ndarray] = []   # each (n_i, K)
        self._starts: list[np.ndarray] = []   # each (n_i, D)
        self._segs:   list[np.ndarray] = []   # each (n_i, D)
        self._scores: list[np.ndarray] = []   # each (n_i,) float32
        self._n_total = 0

    # ------------------------------------------------------------------
    def add(
        self,
        svd_features: np.ndarray,   # (B, K)
        starts:       np.ndarray,   # (B, D)
        segs:         np.ndarray,   # (B, D)
        scores:       np.ndarray,   # (B,)
    ):
        """Add a batch of hard-noise windows to the bank."""
        self._svd.append(svd_features.astype(np.float32))
        self._starts.append(starts.astype(np.int64))
        self._segs.append(segs.astype(np.int64))
        self._scores.append(scores.astype(np.float32))
        self._n_total += len(svd_features)
        if self._n_total > self.max_bank_size:
            self._trim()

    # ------------------------------------------------------------------
    def _trim(self):
        """Prune to capacity keeping top-50% by score + random 50%."""
        all_svd    = np.concatenate(self._svd,    axis=0)
        all_starts = np.concatenate(self._starts, axis=0)
        all_segs   = np.concatenate(self._segs,   axis=0)
        all_scores = np.concatenate(self._scores, axis=0)
        n          = self.max_bank_size
        half       = n // 2
        top_idx    = np.argpartition(all_scores, -half)[-half:]
        rest_mask  = np.ones(len(all_scores), dtype=bool)
        rest_mask[top_idx] = False
        rest_idx = np.where(rest_mask)[0]
        if len(rest_idx) > half:
            rest_idx = np.random.choice(rest_idx, half, replace=False)
        keep = np.concatenate([top_idx, rest_idx])
        self._svd    = [all_svd[keep]]
        self._starts = [all_starts[keep]]
        self._segs   = [all_segs[keep]]
        self._scores = [all_scores[keep]]
        self._n_total = len(keep)

    # ------------------------------------------------------------------
    def _flat(self):
        """Return flat concatenated arrays (svd, starts, segs, scores)."""
        if self._n_total == 0:
            D = 2   # sensible default; will be empty anyway
            return (
                np.empty((0, self.K), dtype=np.float32),
                np.empty((0, D), dtype=np.int64),
                np.empty((0, D), dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )
        return (
            np.concatenate(self._svd,    axis=0),
            np.concatenate(self._starts, axis=0),
            np.concatenate(self._segs,   axis=0),
            np.concatenate(self._scores, axis=0),
        )

    # ------------------------------------------------------------------
    def min_distances(self, batch_svd: np.ndarray) -> np.ndarray:
        """
        Min L2 distance from each row of ``batch_svd`` to the nearest bank
        template.  Returns ``np.inf`` when the bank is empty.
        """
        B = len(batch_svd)
        if self._n_total == 0:
            return np.full(B, np.inf, dtype=np.float32)

        bank, *_ = self._flat()
        if len(bank) > self.max_query_size:
            bank = bank[np.random.choice(len(bank), self.max_query_size, replace=False)]

        bank_norms  = (bank       ** 2).sum(1)
        batch_norms = (batch_svd  ** 2).sum(1)
        cross       = batch_svd @ bank.T
        sq_dists    = batch_norms[:, None] - 2 * cross + bank_norms[None, :]
        return np.sqrt(np.maximum(sq_dists, 0).min(1)).astype(np.float32)

    # ------------------------------------------------------------------
    def sample_starts(self, n: int, rng: np.random.Generator):
        """Sample ``n`` (start, seg) rows uniformly from the bank."""
        if self._n_total == 0:
            return None, None
        _, all_st, all_sg, _ = self._flat()
        idx = rng.integers(0, len(all_st), size=n)
        return all_st[idx], all_sg[idx]

    def sample_scored(self, n: int, rng: np.random.Generator):
        """
        Sample ``n`` rows from the bank, biased toward higher scores (top-half
        preferred).  Returns (svd, starts, segs, scores) all shape (n, *).
        """
        if self._n_total == 0:
            return None, None, None, None
        all_svd, all_st, all_sg, all_sc = self._flat()
        half = max(1, len(all_sc) // 2)
        top_idx  = np.argpartition(all_sc, -half)[-half:]
        if n <= len(top_idx):
            chosen = rng.choice(top_idx, size=n, replace=False)
        else:
            chosen = rng.integers(0, len(all_sc), size=n)
        return all_svd[chosen], all_st[chosen], all_sg[chosen], all_sc[chosen]

    # ------------------------------------------------------------------
    def save(self, path: str | Path):
        """Save bank to a compressed .npz file."""
        p = Path(path)
        if p.suffix != ".npz":
            p = p.with_suffix(".npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        svd, st, sg, sc = self._flat()
        np.savez_compressed(
            str(p),
            svd=svd, starts=st, segs=sg, scores=sc,
            K=np.array([self.K]),
            max_bank_size=np.array([self.max_bank_size]),
        )

    @classmethod
    def load(cls, path: str | Path) -> SharedHardNoiseBank:
        """Load bank from a .npz file saved by ``save``."""
        p = Path(path)
        if p.suffix != ".npz":
            p = p.with_suffix(".npz")
        d    = np.load(str(p), allow_pickle=False)
        K    = int(d["K"][0])
        mbs  = int(d["max_bank_size"][0])
        bank = cls(K=K, max_bank_size=mbs)
        if len(d["scores"]) > 0:
            bank._svd    = [d["svd"]]
            bank._starts = [d["starts"]]
            bank._segs   = [d["segs"]]
            bank._scores = [d["scores"]]
            bank._n_total = len(d["scores"])
        return bank

    def __len__(self):
        return self._n_total


# ---------------------------------------------------------------------------
# CMAMEGAMiner
# ---------------------------------------------------------------------------

class CMAMEGAMiner:
    """
    Template-bank guided hard noise miner.

    Maintains a bank of SVD fingerprints of previously found hard noise
    windows and uses it as a fast pre-filter so the model only scores
    candidates that *look like* noise it has struggled with before.

    Why this finds millions
    -----------------------
    At threshold=5+ the far tail of the score distribution is very sparse.
    Evaluating the model on every random window wastes 99.9 % of GPU time on
    windows that will never score above threshold.  The template bank solves
    this:

    1. Project each random noise window into K-dim SVD space (< 1 ms per
       batch on CPU — purely matrix multiply).
    2. Compute distance to the nearest bank template.
    3. **Only** call the model on windows that are within the bank's
       SVD neighbourhood.

    Hard noise tends to cluster in feature space (glitches recur, loud
    periods persist).  A window that looks like a known hard window is far
    more likely to also score above threshold than a random window.

    Two search modes run in parallel:

    EXPLORE  (fraction = 1 - exploit_fraction)
        Sample random windows from the noise memmap.
        SVD-filter against the bank.  Score candidates that pass.
        Passes windows that are within *svd_distance_pct* percentile of
        distances from random noise to the bank (auto-calibrated each run).
        When the bank is empty (first ever run): all windows pass the filter;
        this is equivalent to BruteForce mining and seeds the bank.

    EXPLOIT  (fraction = exploit_fraction)
        A CMA-ES running in continuous GPS start-time space (same mechanism
        as CMAMEMiner) learns the *covariance* of hard GPS positions in the
        bank — which windows in time cluster together, how wide the cluster
        is, whether H1 and L1 hard times are correlated.  The CMA proposes
        new start times, scores them, and sharpens its distribution toward
        higher-scoring positions.  When it converges on one cluster (or all
        scores drop below threshold), it reinitialises its mean to a randomly
        chosen bank member and explores a new cluster.

        This gives CMAMEGAMiner the same GPS learning power as CMAMEMiner
        while the SVD filter in the explore mode keeps it from wasting model
        calls on obviously easy noise.

    The bank grows during mining: every above-threshold window found by
    either mode is added.  The bank is also seeded from any ``prior_dataset``
    passed to ``mine()`` (the most recent versioned dataset from the previous
    run), giving the filter immediate effectiveness from iteration 1.

    Parameters
    ----------
    n_svd_components : int  (K)
        SVD latent dimension used for fingerprinting.
    n_init_batches : int
        Random batches used to fit the SVD projector before mining begins.
        Does NOT need to be large — just enough to calibrate the SVD basis.
    n_iterations : int
        Total mining iterations.  Each iteration reads ``scan_batch_size``
        windows.  Increase this to collect more samples.
        ``target_samples`` provides an alternative stop condition.
    scan_batch_size : int
        Windows read from the noise file per iteration (large batch →
        more candidates for the SVD filter before a model call).
    model_batch_size : int
        Max windows forwarded through the model in one call.
    exploit_fraction : float
        Fraction of iterations devoted to GPS-CMA exploitation.
    sigma_g_gps : float
        Initial CMA step size in logit-space GPS coordinates for the exploit
        CMA.  Same meaning as ``sigma_g`` in CMAMEMiner.
    min_exploit_count : int
        Minimum exploit CMA steps before reinitialisation is allowed.
    svd_distance_pct : float
        SVD filter threshold percentile.  The filter passes windows whose
        distance to the nearest bank template is below the ``svd_distance_pct``
        percentile of random-noise-to-bank distances.  Lower → tighter
        filter, fewer model calls, higher yield per call.
        Typical range: 5–25.
    threshold : float
        Minimum model ranking statistic to save a window.
    target_samples : int or None
        Stop early once this many above-threshold samples have been
        accumulated.  ``None`` runs for all ``n_iterations``.
    max_samples : int
        Hard cap on accumulated samples (streaming top-K prune).
    max_bank_size : int
        Maximum number of templates kept in the bank for NN lookup.
    autocast : bool
        Float16 AMP during model inference.
    """

    def __init__(
        self,
        n_svd_components:    int   = 32,
        n_init_batches:      int   = 200,
        n_iterations:        int   = 50_000,
        scan_batch_size:     int   = 512,
        model_batch_size:    int   = 128,
        exploit_fraction:    float = 0.3,
        pure_explore_frac:   float = 0.1,
        sigma_g_gps:         float = 0.5,
        min_exploit_count:   int   = 10,
        svd_distance_pct:    float = 10.0,
        refine_lr:           float = 1.0,
        n_rescore:           int   = 2_000,
        threshold:           float = 5.0,
        target_samples:      int | None = None,
        max_samples:         int   = 5_000_000,
        max_bank_size:       int   = 100_000,
        autocast:            bool  = True,
    ):
        self.n_svd_components   = n_svd_components
        self.n_init_batches     = n_init_batches
        self.n_iterations       = n_iterations
        self.scan_batch_size    = scan_batch_size
        self.model_batch_size   = model_batch_size
        self.exploit_fraction   = exploit_fraction
        self.pure_explore_frac  = pure_explore_frac
        self.sigma_g_gps        = sigma_g_gps
        self.min_exploit_count  = min_exploit_count
        self.svd_distance_pct   = svd_distance_pct
        self.refine_lr          = refine_lr
        self.n_rescore          = n_rescore
        self.threshold          = threshold
        self.target_samples     = target_samples
        self.max_samples        = max_samples
        self.max_bank_size      = max_bank_size
        self.autocast           = autocast

    # ------------------------------------------------------------------
    @staticmethod
    def _score_batch(model, x: torch.Tensor, cast) -> np.ndarray:
        """Forward pass; returns (B,) float32 scores."""
        with torch.no_grad(), cast:
            out = model(x)
        return out[0].squeeze(1).float().cpu().numpy()

    # ------------------------------------------------------------------
    def mine(
        self,
        model,
        noise_sampler,
        processor,
        device: str,
        signal_sampler=None,
        shared_bank: SharedHardNoiseBank | None = None,
    ) -> StartTimeDataset:
        """
        Run the template-bank mining pass.

        Parameters
        ----------
        model : nn.Module
        noise_sampler : MemmapNoiseSampler
        processor : nn.Module
        device : str
        signal_sampler : nn.Module or None
            Always pass this for correct GWBatch preprocessing.
        shared_bank : SharedHardNoiseBank or None
            Shared knowledge bank.  When provided:
            - All existing templates are re-evaluated with the current model.
            - Stale templates are gradient-refined toward harder real noise.
            - All new findings are added in-place.
            - Bank SVD templates serve as the explore pre-filter from iter 1.
            - Bank GPS positions seed the exploit GPS CMA immediately.
            Pass the same bank to CMAMEMiner so both miners share knowledge.

        Returns
        -------
        StartTimeDataset
            Every window above ``threshold`` found this run, score-sorted.
        """
        reader        = _MiningReader(noise_sampler)
        was_training  = model.training
        model.eval()
        preprocess_fn = make_miner_preprocessor(processor, signal_sampler)

        cast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.autocast else nullcontext()
        )

        rng  = np.random.default_rng()
        proj = NoiseSVDProjector(n_components=self.n_svd_components, pca_dims=2)
        accum = _StreamingAccumulator(max_samples=self.max_samples)

        # Use shared bank if provided; otherwise create a local ephemeral one
        bank: SharedHardNoiseBank = (
            shared_bank
            if shared_bank is not None
            else SharedHardNoiseBank(K=self.n_svd_components, max_bank_size=self.max_bank_size)
        )

        # GPS CMA for exploit mode
        gps_mapper = _GenotypeMapper(reader.seg_index, reader.seq_len)
        gps_cmaes  = _CMAES(
            search_dim      = reader.n_detectors,
            population_size = self.scan_batch_size,
            init_sigma      = self.sigma_g_gps,
        )

        # ── Phase 0: Build random pool + fit SVD projector ───────────────
        # Pool is retained for NN lookup during gradient refinement (Phase 1).
        print(f"[CMA-MEGA] Building pool: {self.n_init_batches} batches …")
        pool_x_list, pool_st_list, pool_sg_list = [], [], []
        for _ in tqdm(range(self.n_init_batches), desc="pool", leave=False):
            st, sg   = reader.random_starts(self.scan_batch_size)
            nfd      = reader.read_batch(st, sg)
            x        = preprocess_fn(nfd)
            pool_x_list.append(x.float().cpu())
            pool_st_list.append(st)
            pool_sg_list.append(sg)
        pool_x  = torch.cat(pool_x_list, 0)
        pool_st = np.concatenate(pool_st_list, 0)
        pool_sg = np.concatenate(pool_sg_list, 0)
        del pool_x_list, pool_st_list, pool_sg_list

        print("[CMA-MEGA] Fitting SVD projector …")
        proj.fit(pool_x)
        pool_svd = proj.svd_encode(pool_x)   # (N_pool, K) for NN lookup

        # ── Phase 1: Re-evaluate + gradient-refine bank templates ─────────
        n_fresh = n_refined = n_removed = 0
        if len(bank) > 0 and self.n_rescore > 0:
            n_eval = min(self.n_rescore, len(bank))
            print(f"[CMA-MEGA] Re-evaluating {n_eval:,} bank templates …")
            bsvd, bst, bsg, _ = bank.sample_scored(n_eval, rng)

            if bsvd is not None:
                fresh_svd, fresh_st, fresh_sg, fresh_sc = [], [], [], []
                chunk = self.model_batch_size

                for b0 in range(0, n_eval, chunk):
                    sl    = slice(b0, min(b0 + chunk, n_eval))
                    s_st  = bst[sl];  s_sg = bsg[sl]
                    nfd   = reader.read_batch(s_st, s_sg)
                    xb    = preprocess_fn(nfd)
                    with torch.no_grad(), cast:
                        sc_b = model(xb)[0].squeeze(1).float().cpu().numpy()

                    still_fresh = sc_b >= self.threshold
                    newly_stale = ~still_fresh

                    if still_fresh.any():
                        svd_b = proj.svd_encode(xb[still_fresh].float().cpu())
                        fresh_svd.append(svd_b);  fresh_st.append(s_st[still_fresh])
                        fresh_sg.append(s_sg[still_fresh]);  fresh_sc.append(sc_b[still_fresh])
                        n_fresh += int(still_fresh.sum())

                    # Gradient refinement for stale templates
                    if newly_stale.any():
                        xstale = preprocess_fn(
                            reader.read_batch(s_st[newly_stale], s_sg[newly_stale])
                        ).to(device).requires_grad_(True)
                        model(xstale)[0].squeeze(1).sum().backward()
                        grads = xstale.grad.detach().cpu()   # (n_st, C, T)

                        for j in range(grads.shape[0]):
                            gf   = grads[j].flatten().numpy().astype(np.float64)
                            gsvd = proj.svd_components.astype(np.float64) @ gf  # (K,)
                            gsvd /= (np.linalg.norm(gsvd) + 1e-8)

                            # SVD of the stale template we're refining
                            stale_local = newly_stale.nonzero()[0][j]
                            old_svd = bsvd[b0 + stale_local].astype(np.float64)
                            svd_target = (old_svd + self.refine_lr * gsvd).astype(np.float32)

                            # Nearest real noise in pool to this target
                            sq  = ((pool_svd - svd_target) ** 2).sum(1)
                            nn  = int(sq.argmin())

                            nfd_nn = reader.read_batch(pool_st[[nn]], pool_sg[[nn]])
                            x_nn   = preprocess_fn(nfd_nn)
                            with torch.no_grad(), cast:
                                sc_nn = float(model(x_nn)[0].squeeze().item())

                            if sc_nn >= self.threshold:
                                svd_nn = proj.svd_encode(x_nn.float().cpu())
                                fresh_svd.append(svd_nn)
                                fresh_st.append(pool_st[[nn]])
                                fresh_sg.append(pool_sg[[nn]])
                                fresh_sc.append(np.array([sc_nn], dtype=np.float32))
                                accum.add(pool_st[[nn]], pool_sg[[nn]],
                                          np.array([sc_nn]), self.threshold)
                                n_refined += 1
                            else:
                                n_removed += 1

                if fresh_svd:
                    bank.add(
                        np.concatenate(fresh_svd), np.concatenate(fresh_st),
                        np.concatenate(fresh_sg),  np.concatenate(fresh_sc),
                    )

            print(
                f"[CMA-MEGA] Bank re-eval: {n_fresh} fresh | "
                f"{n_refined} gradient-refined to harder | "
                f"{n_removed} discarded (model mastered)"
            )

        print(f"[CMA-MEGA] Bank: {len(bank):,} templates")

        # ── Calibrate SVD distance threshold ─────────────────────────────
        svd_dist_threshold = np.inf
        if len(bank) > 0:
            cal_svds  = [proj.svd_encode(pool_x[i:i+self.scan_batch_size])
                         for i in range(0, min(len(pool_x), 10 * self.scan_batch_size),
                                        self.scan_batch_size)]
            cal_all   = np.concatenate(cal_svds[:20])
            cal_dists = bank.min_distances(cal_all)
            svd_dist_threshold = float(np.percentile(cal_dists, self.svd_distance_pct))
            pct_pass  = float((cal_dists <= svd_dist_threshold).mean()) * 100
            print(
                f"[CMA-MEGA] SVD filter: threshold={svd_dist_threshold:.4f} "
                f"({pct_pass:.1f}% of random windows pass)"
            )

        # ── Init GPS CMA from bank ────────────────────────────────────────
        init_st, init_sg = bank.sample_starts(1, rng) if len(bank) > 0 else (None, None)
        gps_cma_state = gps_cmaes.init(
            mean=gps_mapper.encode(init_st, init_sg)[0] if init_st is not None else None
        )
        exploit_emit_count = 0

        n_explored = n_exploited = n_passed_svd = n_model_calls = 0
        log_every = max(1, self.n_iterations // 20)

        # ── Phase 2: Main mining loop ─────────────────────────────────────
        print(f"[CMA-MEGA] Mining: {self.n_iterations} iterations …")

        for i in tqdm(range(self.n_iterations), desc="CMA-MEGA", leave=False):

            use_exploit = len(bank) > 0 and rng.random() < self.exploit_fraction

            if use_exploit:
                # ── Exploit: GPS CMA (learns WHERE hard noise is) ─────────
                candidates   = gps_cmaes.sample(gps_cma_state, rng)
                starts, segs = gps_mapper.decode(candidates)
                noise_fd     = reader.read_batch(starts, segs)
                x            = preprocess_fn(noise_fd)
                scores       = self._score_batch(model, x.to(device), cast)
                n_model_calls += len(scores);  n_exploited += len(scores)

                sorted_idx    = np.argsort(scores)[::-1]
                gps_cma_state = gps_cmaes.update(gps_cma_state, candidates[sorted_idx])
                exploit_emit_count += 1

                above = scores >= self.threshold
                if above.any():
                    svd_ab = proj.svd_encode(x[above].float().cpu())
                    bank.add(svd_ab, starts[above], segs[above], scores[above])
                accum.add(starts, segs, scores, self.threshold)

                stuck = (
                    exploit_emit_count > self.min_exploit_count
                    and np.all(scores < self.threshold)
                ) or gps_cmaes.stop_condition(gps_cma_state)
                if stuck:
                    new_st, new_sg = bank.sample_starts(1, rng)
                    mean = gps_mapper.encode(new_st, new_sg)[0] if new_st is not None else None
                    gps_cma_state  = gps_cmaes.init(mean=mean)
                    exploit_emit_count = 0

            else:
                # ── Explore: random scan, two sub-modes ──────────────────
                starts, segs = reader.random_starts(self.scan_batch_size)
                noise_fd     = reader.read_batch(starts, segs)
                x            = preprocess_fn(noise_fd)
                n_explored  += len(starts)
                svd_batch    = proj.svd_encode(x.float().cpu())

                # Sub-mode A: pure random (pure_explore_frac) — no SVD filter
                # Finds genuinely novel noise types not yet in the bank.
                use_pure = rng.random() < self.pure_explore_frac or len(bank) == 0

                if use_pure:
                    pass_mask = np.ones(len(svd_batch), dtype=bool)
                else:
                    # Sub-mode B: SVD-filtered — only score near-bank candidates
                    dists     = bank.min_distances(svd_batch)
                    pass_mask = dists <= svd_dist_threshold

                n_passed_svd += int(pass_mask.sum())
                if not pass_mask.any():
                    continue

                cands_x = x[pass_mask].to(device)
                all_sc  = []
                for b0 in range(0, len(cands_x), self.model_batch_size):
                    all_sc.append(self._score_batch(model, cands_x[b0:b0+self.model_batch_size], cast))
                scores = np.concatenate(all_sc)
                n_model_calls += len(scores)

                above = scores >= self.threshold
                if above.any():
                    cand_svd = svd_batch[pass_mask][above]
                    cand_st  = starts[pass_mask][above]
                    cand_sg  = segs[pass_mask][above]
                    bank.add(cand_svd, cand_st, cand_sg, scores[above])
                    # Periodically recalibrate SVD filter as bank grows
                    if (i + 1) % 500 == 0:
                        new_dists = bank.min_distances(svd_batch)
                        svd_dist_threshold = float(
                            np.percentile(new_dists, self.svd_distance_pct)
                        )
                accum.add(starts[pass_mask], segs[pass_mask], scores, self.threshold)

            if (i + 1) % log_every == 0:
                print(
                    f"  [iter {i+1:,}/{self.n_iterations}] "
                    f"bank: {len(bank):,} | accumulated: {len(accum):,} | "
                    f"model calls: {n_model_calls:,} | "
                    f"SVD pass: {100*n_passed_svd/max(n_explored,1):.1f}%"
                )

            if self.target_samples is not None and len(accum) >= self.target_samples:
                print(f"[CMA-MEGA] target_samples={self.target_samples:,} reached.")
                break

        if was_training:
            model.train()

        dataset = accum.to_dataset(noise_sampler, reader)
        print(
            f"[CMA-MEGA] Done — {len(dataset):,} windows above {self.threshold:.3f} | "
            f"best: {'n/a' if len(dataset)==0 else f'{dataset.scores.max():.3f}'} | "
            f"model calls: {n_model_calls:,} | "
            f"yield: {100*len(dataset)/max(n_model_calls,1):.2f}%"
        )
        return dataset
