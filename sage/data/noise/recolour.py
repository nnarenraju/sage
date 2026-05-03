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
import os
import json
import torch
import numpy as np

from pathlib import Path

# LOCAL
from sage.core.config import get_cfg, get_data_cfg


class RecolourPostprocess(torch.nn.Module):
    """
    GPU postprocessing step: stochastic PSD recolouring from one noise epoch
    to another, operating entirely in the frequency domain.

    Motivation
    ----------
    Sage trains on O3b noise but evaluates on O3a noise (different GPS epoch,
    different spectral shape).  Simply whitening with O3b PSDs and testing on
    O3a produces a distribution shift.  With ``p_recolour`` probability, each
    training sample is:

    1. **Whitened** using the segment's own O3b ASD (removing O3b colour).
    2. **Recoloured** by multiplying with a randomly chosen O3a ASD (adding
       O3a colour).

    The remaining ``1 - p_recolour`` fraction of the batch passes through
    unchanged (plain FD conversion only).

    This bridges the spectral gap between training and evaluation epochs
    without using any actual O3a time-domain data during training.  Note
    that glitch *morphology* is not altered — only the spectral amplitude
    envelope changes.

    Parameters
    ----------
    p_recolour : float in [0, 1]
        Per-sample probability of applying the whiten + recolour transform.
        Typical value: 0.37.
    recolour_dataset_dir : str
        Root directory of the *target* noise epoch dataset (e.g. the O3a
        data release directory).  Must contain a ``data_dir/recolour_psds/``
        sub-directory with pre-computed per-detector ASD banks.
    eps : float
        Small value added to ASDs before division/multiplication to prevent
        division by zero in very quiet frequency bins.

    Inputs / Outputs
    ----------------
    forward(batch_td, segment_ids) :
        ``batch_td``  : ``(B, D, T)`` float32 — time-domain noise windows.
        ``segment_ids``: ``(B, D)`` int64 — index into the segment ASD bank
        (used to select the correct per-segment whitening ASD).

    Returns ``(B, D, F)`` complex64 — frequency-domain (recoloured) strain.
    """

    def __init__(
        self,
        *,
        p_recolour: float,
        recolour_dataset_dir: str,
        eps: float = 1e-38,
    ):
        super().__init__()

        # Setup configs
        cfg = get_cfg()
        data_cfg = get_data_cfg()

        self.data_dir = Path(data_cfg.data_dir)
        self.recolour_dataset_dir = Path(recolour_dataset_dir)
        self.detectors = cfg.detectors
        self.seq_len = data_cfg.padded_length_in_nsamples
        self.sample_rate = data_cfg.sample_rate
        self.p_recolour = float(p_recolour)
        self.device = cfg.device
        self.eps = eps

        self.B = cfg.batch_size
        self.D = len(self.detectors)

        # We expect this length from the PSDs
        # Interpolate them after production
        self.n_freq = self.seq_len // 2 + 1

        # Load PSDs to torch.float32 on GPU
        self._load_segment_asds()
        self._load_recolour_asds()

    def _load_segment_asds(self):
        """
        Loads pre-interpolated per-segment ASDs.
        We need to pad the bank to a rectangular format
        This supports tensor ops downstream

        Result:
            self.segment_asds: (D, N_seg_max, F) float32
        """
        asds_per_det = []
        max_nseg = 0

        for det in self.detectors:
            # Segment ASDs should be from the noise used for training
            asd_dir = self.data_dir / "segment_psds"
            bin_path = asd_dir / f"data_{det}_psds.bin"
            meta_path = asd_dir / f"data_{det}_psds_segments.json"

            with open(meta_path, "r") as f:
                meta = json.load(f)

            raw = np.fromfile(bin_path, dtype=np.float32)

            asds = []
            cursor = 0
            for m in meta:
                n = m["psd_len"]  # should already be == F
                asds.append(raw[cursor : cursor + n])
                cursor += n

            asds = np.stack(asds, axis=0)  # (N_seg, F)
            asds_per_det.append(asds)
            max_nseg = max(max_nseg, asds.shape[0])

        # Pad to rectangular (D, N_seg_max, F)
        padded = []
        for asds in asds_per_det:
            pad = max_nseg - asds.shape[0]
            if pad > 0:
                asds = np.pad(asds, ((0, pad), (0, 0)), constant_values=1.0)
            padded.append(asds)

        self.segment_asds = torch.from_numpy(np.stack(padded, axis=0))

    def _load_recolour_asds(self):
        """
        Loads recolour ASD bank (already interpolated).
        Bank size should be the same for all dets;
        So we don't need to pad this to form a rectangular tensor

        Result:
            self.recolour_asds: (D, N_asd, F) float32
        """
        asds_all = []

        for det in self.detectors:
            # Recolour ASDs can be different from that for training
            data_dir = Path(os.path.join(self.recolour_dataset_dir, "data_dir"))
            asd_dir = data_dir / "recolour_psds"
            bin_path = asd_dir / f"raw_{det}_psds.bin"
            meta_path = asd_dir / f"raw_{det}_psds.json"

            with open(meta_path, "r") as f:
                meta = json.load(f)

            n_asd = meta["num_psds"]
            n_freq = meta["num_freq_bins"]  # should == F

            asds = np.fromfile(bin_path, dtype=np.float32).reshape(n_asd, n_freq)
            asds_all.append(asds)

        self.recolour_asds = torch.from_numpy(np.stack(asds_all, axis=0))
        self.n_recolour_asd = self.recolour_asds.shape[1]

    @torch.no_grad()
    def forward(
        self,
        batch_td: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        torch.compile-safe FD recolouring
        """

        # TD to FD (B, D, F)
        X = torch.fft.rfft(batch_td, dim=-1, norm="forward")

        # Bernoulli recolour mask (B, D, 1)
        # RuntimeError: Offset increment outside graph capture encountered unexpectedly.
        # mask = torch.rand(self.B, self.D, 1, device=X.device) < self.p_recolour

        mask_cpu = torch.rand(self.B, self.D, 1) < self.p_recolour
        mask = mask_cpu.to(X.device, non_blocking=True)

        # Whitening PSD (only where mask == True, else ones)
        det_idx = torch.arange(self.D).view(1, self.D).expand(self.B, self.D)
        gathered_seg_asd = self.segment_asds[det_idx, segment_ids]
        gathered_seg_asd = gathered_seg_asd.to(X.device, non_blocking=True)

        X = torch.where(
            mask,
            X / (gathered_seg_asd + self.eps),
            X,
        )

        # Recolour PSD (only where mask == True)
        recol_idx = torch.randint(0, self.n_recolour_asd, (self.B, self.D))
        gathered_recol_asd = self.recolour_asds[det_idx, recol_idx]
        gathered_recol_asd = gathered_recol_asd.to(X.device)

        recol_gain = gathered_recol_asd + self.eps

        X = torch.where(mask, X * recol_gain, X)

        return X
