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

import matplotlib.pyplot as plt


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
    GPU postprocessing: conditional whitening + stochastic recolouring in FD.

    Inputs:
        batch_td:   (B, D, T) float32
        segment_ids:(B, D) int32

    Output:
        batch_fd:   (B, D, F) complex64
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
        eps: float = 1e-60,
    ):
        self.data_dir = Path(data_dir)
        self.detectors = detectors
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.p_recolour = float(p_recolour)
        self.device = device
        self.eps = eps

        self.n_detectors = len(detectors)

        # We expect this length from the PSDs
        # Interpolate them after production
        self.n_freq = seq_len // 2 + 1

        # Load PSDs to torch.float32 on GPU
        self._load_segment_psds()
        self._load_recolour_psds()

    def _load_segment_psds(self):
        """
        Loads pre-interpolated per-segment PSDs.
        We need to pad the bank to a rectangular format
        This supports tensor ops downstream

        Result:
            self.segment_psds: (D, N_seg_max, F) float32
        """
        psds_per_det = []
        max_nseg = 0

        for det in self.detectors:
            psd_dir = self.data_dir / "segment_psds"
            bin_path = psd_dir / f"data_{det}_O3a_psds.bin"
            meta_path = psd_dir / f"data_{det}_O3a_psds_segments.json"

            with open(meta_path, "r") as f:
                meta = json.load(f)

            raw = np.fromfile(bin_path, dtype=np.float32)

            psds = []
            cursor = 0
            for m in meta:
                n = m["psd_len"]  # should already be == F
                psds.append(raw[cursor : cursor + n])
                cursor += n

            psds = np.stack(psds, axis=0)  # (N_seg, F)
            psds_per_det.append(psds)
            max_nseg = max(max_nseg, psds.shape[0])

        # Pad to rectangular (D, N_seg_max, F)
        padded = []
        for psds in psds_per_det:
            pad = max_nseg - psds.shape[0]
            if pad > 0:
                psds = np.pad(psds, ((0, pad), (0, 0)), constant_values=1.0)
            padded.append(psds)

        self.segment_psds = torch.from_numpy(np.stack(padded, axis=0)).float()

    def _load_recolour_psds(self):
        """
        Loads recolour PSD bank (already interpolated).
        Bank size should be the same for all dets;
        So we don't need to pad this to form a rectangular tensor

        Result:
            self.recolour_psds: (D, N_psd, F) float32
        """
        psds_all = []

        for det in self.detectors:
            psd_dir = self.data_dir / "recolour_psds"
            bin_path = psd_dir / f"raw_{det}_psds.bin"
            meta_path = psd_dir / f"raw_{det}_psds.json"

            with open(meta_path, "r") as f:
                meta = json.load(f)

            n_psd = meta["num_psds"]
            n_freq = meta["num_freq_bins"]  # should == F

            psds = np.fromfile(bin_path, dtype=np.float32).reshape(n_psd, n_freq)
            psds_all.append(psds)

        self.recolour_psds = torch.from_numpy(np.stack(psds_all, axis=0)).float()
        self.n_recolour_psd = self.recolour_psds.shape[1]

    def __call__(
        self,
        batch_td: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        torch.compile-safe FD recolouring
        """
        B, D, _ = batch_td.shape

        # TD to FD
        X = torch.fft.rfft(batch_td, dim=-1)  # (B, D, F)

        # Bernoulli recolour mask
        mask = torch.rand(B, D, 1, device=X.device) < self.p_recolour  # (B, D, 1)

        # Whitening PSD (only where mask == True, else ones)
        det_idx = torch.arange(D).view(1, D).expand(B, D)
        gathered_seg_psd = self.segment_psds[det_idx, segment_ids]
        gathered_seg_psd = gathered_seg_psd.to(X.device, non_blocking=True)

        X = torch.where(
            mask,
            X / torch.sqrt(gathered_seg_psd + self.eps),
            X,
        )

        # Recolour PSD (only where mask == True)
        recol_idx = torch.randint(0, self.n_recolour_psd, (B, D))
        gathered_recol_psd = self.recolour_psds[det_idx, recol_idx]
        gathered_recol_psd = gathered_recol_psd.to(X.device)

        recol_gain = torch.sqrt(gathered_recol_psd + self.eps)

        X = torch.where(mask, X * recol_gain, X)

        return X
