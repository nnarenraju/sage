#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : recolour.py
Description     : Short description of the file

Created on 2026-02-09 23:39:57

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


class FFTOnlyPostprocess:
    """
    GPU postprocessing: time-domain → frequency-domain only.

    Inputs:
        batch_td: (B, D, T) float32

    Output:
        batch_fd: (B, D, F) complex64
    """

    def __init__(
        self,
        *,
        seq_len: int,
        device: str = "cuda",
    ):
        self.seq_len = seq_len
        self.device = device

        # Optional sanity check
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

    def __call__(
        self,
        batch_td: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert batch of TD noise to FD.

        Args:
            batch_td: (B, D, T) float32 tensor on GPU

        Returns:
            batch_fd: (B, D, F) complex64 tensor on GPU
        """
        # Sanity checks (cheap, compile-safe)
        if batch_td.ndim != 3:
            raise ValueError("batch_td must have shape (B, D, T)")

        if batch_td.shape[-1] != self.seq_len:
            raise ValueError(f"Expected T={self.seq_len}, got {batch_td.shape[-1]}")

        # Time domain to frequency domain
        batch_fd = torch.fft.rfft(batch_td, dim=-1)

        return batch_fd


class RecolourPostprocess:
    """
    GPU postprocessing: whitening + stochastic recolouring in FD.

    Inputs:
        batch_td: (B, D, T) float32
        segment_ids: (B, D) int32

    Output:
        batch_fd: (B, D, F) complex64
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        detectors: list[str],
        seq_len: int,
        sample_rate: float,
        p_recolour: float,
        device: str = "cuda",
        eps: float = 1e-12,
    ):
        self.data_dir = Path(data_dir)
        self.detectors = detectors
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.p_recolour = float(p_recolour)
        self.device = device
        self.eps = eps

        self.n_detectors = len(detectors)

        self.delta_f_psd = None
        self.psd_freqs = None

        self.delta_f_fft = 1.0 / (self.seq_len / self.sample_rate)
        self.fft_freqs = (
            torch.arange(self.seq_len // 2 + 1, device=self.device) * self.delta_f_fft
        )

        # Load PSDs (CPU --> pinned --> GPU once)
        self._load_segment_psds()
        self._load_recolour_psds()

    def _load_segment_psds(self):
        """
        Loads per-segment PSDs into GPU tensors.

        Result:
            self.segment_psds[d][seg_idx] -> (F,) float64
        """
        self.segment_psds = []

        for det in self.detectors:
            psd_dir = self.data_dir / "segment_psds"
            bin_path = psd_dir / f"data_{det}_psds.bin"
            meta_path = psd_dir / f"data_{det}_psds_segments.json"

            with open(meta_path, "r") as f:
                meta = json.load(f)

            # Read entire bin
            psds = np.fromfile(bin_path, dtype=np.float64)

            psd_list = []
            cursor = 0

            for m in meta:
                n = m["psd_len"]
                psd = psds[cursor : cursor + n]
                cursor += n
                psd_list.append(psd)

            self.segment_psds.append(np.stack(psd_list, axis=0))

    def _load_recolour_psds(self):
        """
        Loads recolour PSD bank.

        Result:
            self.recolour_psds[d] -> (N_psd, F)
        """
        self.recolour_psds = []

        for det in self.detectors:
            psd_dir = self.data_dir / "recolour_psds"
            bin_path = psd_dir / f"raw_{det}_psds.bin"
            meta_path = psd_dir / f"raw_{det}_psds.json"

            with open(meta_path, "r") as f:
                meta = json.load(f)

            n_psd = meta["num_psds"]
            n_freq = meta["num_freq_bins"]

            psds = np.fromfile(bin_path, dtype=np.float64)
            psds = psds.reshape(n_psd, n_freq)

            self.recolour_psds.append(psds)

    def __call__(
        self,
        batch_td: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Whitening + stochastic recolouring in FD (torch.compile safe)
        """
        B, D, T = batch_td.shape

        # TD → FD
        X = torch.fft.rfft(batch_td, dim=-1)  # (B, D, F)
        F = X.shape[-1]

        # Whitening
        # Gather PSDs: (B, D, F)
        psd_whiten = torch.gather(
            self.segment_psds, dim=1, index=segment_ids.unsqueeze(-1).expand(-1, -1, F)
        ).transpose(0, 1)

        X = X / torch.sqrt(psd_whiten + self.eps)

        # Stochastic recolouring
        # Mask: (B, D) to (B, D, 1)
        recolour_mask = torch.rand(B, D, device=X.device) < self.p_recolour
        recolour_mask = recolour_mask.unsqueeze(-1)

        # Sample PSD indices for *all* samples (cheap, avoids branching)
        N_psd = self.recolour_psds.shape[1]
        psd_idx = torch.randint(0, N_psd, (B, D), device=X.device)

        # Gather recolour PSDs: (B, D, F)
        psd_recolour = torch.gather(
            self.recolour_psds, dim=1, index=psd_idx.unsqueeze(-1).expand(-1, -1, F)
        ).transpose(0, 1)

        recolour_gain = torch.sqrt(psd_recolour + self.eps)

        # Apply only where masked
        X = torch.where(recolour_mask, X * recolour_gain, X)

        return X
