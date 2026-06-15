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
        Lazily memory-map the per-segment ASD banks (one per detector). Rows are
        gathered on demand in :meth:`forward` (see :meth:`_gather`) so the full
        banks are never resident — only the rows a batch touches are read.

        No rectangular padding is needed: each detector is indexed into its own
        bank, and after the dense ``segment_index`` renumbering the sampler emits
        per-detector positional ids in ``[0, N_seg_d)``.

        Result:
            self._segment_mm: list[np.memmap]  per detector, shape (N_seg_d, F)
        """
        self._segment_mm = []

        for det in self.detectors:
            # Segment ASDs should be from the noise used for training
            asd_dir = self.data_dir / "segment_psds"
            bin_path = asd_dir / f"data_{det}_psds.bin"
            meta_path = asd_dir / f"data_{det}_psds_segments.json"

            with open(meta_path, "r") as f:
                meta = json.load(f)

            n_seg = len(meta)
            n_freq = int(meta[0]["psd_len"])  # interpolated to a fixed F
            expected = n_seg * n_freq * 4
            actual = os.path.getsize(bin_path)
            if actual != expected:
                raise ValueError(
                    f"segment ASD bank {bin_path}: size {actual} != expected "
                    f"{expected} ({n_seg} x {n_freq} float32) — non-uniform psd_len?"
                )

            mm = np.memmap(bin_path, dtype=np.float32, mode="r", shape=(n_seg, n_freq))
            self._segment_mm.append(mm)

    def _load_recolour_asds(self):
        """
        Lazily memory-map the recolour ASD bank (one per detector). Rows are
        gathered on demand in :meth:`forward`; the ~16 GB/detector banks never
        need to be resident.

        Result:
            self._recolour_mm: list[np.memmap]  per detector, shape (N_asd, F)
            self.n_recolour_asd: int
        """
        self._recolour_mm = []
        self.n_recolour_asd = None

        for det in self.detectors:
            # Recolour ASDs can be different from that for training
            data_dir = Path(os.path.join(self.recolour_dataset_dir, "data_dir"))
            asd_dir = data_dir / "recolour_psds"
            bin_path = asd_dir / f"raw_{det}_psds.bin"
            meta_path = asd_dir / f"raw_{det}_psds.json"

            with open(meta_path, "r") as f:
                meta = json.load(f)

            n_asd = int(meta["num_psds"])
            n_freq = int(meta["num_freq_bins"])  # should == F
            expected = n_asd * n_freq * 4
            actual = os.path.getsize(bin_path)
            if actual != expected:
                raise ValueError(
                    f"recolour ASD bank {bin_path}: size {actual} != expected "
                    f"{expected} ({n_asd} x {n_freq} float32)"
                )

            mm = np.memmap(bin_path, dtype=np.float32, mode="r", shape=(n_asd, n_freq))
            self._recolour_mm.append(mm)

            if self.n_recolour_asd is None:
                self.n_recolour_asd = n_asd

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
        mask_cpu = torch.rand(self.B, self.D, 1) < self.p_recolour
        mask = mask_cpu.to(X.device, non_blocking=True)

        # Whitening PSD: each window's own segment ASD, gathered lazily per det
        seg_idx = segment_ids.detach().cpu().numpy()
        gathered_seg_asd = self._gather(self._segment_mm, seg_idx)
        gathered_seg_asd = gathered_seg_asd.to(X.device, non_blocking=True)

        X = torch.where(
            mask,
            X / (gathered_seg_asd + self.eps),
            X,
        )

        # Recolour PSD: a random ASD from the target-epoch bank (lazy gather)
        recol_idx = torch.randint(0, self.n_recolour_asd, (self.B, self.D)).numpy()
        gathered_recol_asd = self._gather(self._recolour_mm, recol_idx)
        gathered_recol_asd = gathered_recol_asd.to(X.device)

        recol_gain = gathered_recol_asd + self.eps

        X = torch.where(mask, X * recol_gain, X)

        return X

    def _gather(self, mmaps, idx):
        """Gather a (B, D, F) ASD tensor from per-detector memmaps.

        Only the rows named in ``idx`` (shape (B, D), integer) are read from each
        detector's bank, so the full banks stay on disk / in the OS page cache
        instead of resident in this process.
        """
        F = mmaps[0].shape[1]
        out = np.empty((self.B, self.D, F), dtype=np.float32)
        for d in range(self.D):
            out[:, d, :] = mmaps[d][idx[:, d]]
        return torch.from_numpy(out)
