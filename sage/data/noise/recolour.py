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

        # Resolve the SAME run set (and ordering) the training noise sampler uses,
        # so recolour's per-run segment banks line up with the run ids the
        # sampler emits. Single-run training -> one run (id 0).
        from sage.data.noise.real_noise import MemmapNoiseSampler
        self._runs = MemmapNoiseSampler._resolve_runs(cfg, data_cfg, training=True)

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
        Load the per-segment ASD banks into RAM, one array per (run, detector).
        They are gathered every batch in :meth:`forward`; on the NFS data mount a
        random per-batch gather costs ~600 ms (vs microseconds from RAM), which
        starves the GPU — so the banks (a few GB total) are kept resident.

        Each detector is read into its own array (no rectangular padding): after
        the dense ``segment_index`` renumbering the sampler emits per-detector
        positional ids in ``[0, N_seg_d)``, keyed within a run. With multiple
        pooled runs the whitening PSD is keyed by (run_id, segment_index), so a
        separate bank is kept per run.

        Result:
            self._segment_banks: list[list[np.ndarray]]  [run_id][d] -> (N_seg, F)
        """
        self._segment_banks = []

        for run in self._runs:
            data_dir = Path(run["data_dir"])
            det_banks = []
            for det in self.detectors:
                # Segment ASDs are from THIS run's training noise.
                asd_dir = data_dir / "segment_psds"
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

                # Read straight into a pre-allocated buffer (no intermediate copy).
                bank = np.empty((n_seg, n_freq), dtype=np.float32)
                with open(bin_path, "rb") as fh:
                    nread = fh.readinto(memoryview(bank).cast("B"))
                if nread != expected:
                    raise ValueError(
                        f"segment ASD bank {bin_path}: read {nread} of {expected} bytes"
                    )
                det_banks.append(bank)
            self._segment_banks.append(det_banks)

    def _load_recolour_asds(self):
        """
        Load the recolour ASD bank into RAM (one array per detector). Each
        detector is read straight into its own pre-allocated buffer — no
        ``list + np.stack`` (which would transiently double the ~16 GB/detector
        banks). Kept resident because a per-batch random gather on the NFS mount
        is ~600 ms and would bottleneck training.

        Result:
            self._recolour_banks: list[np.ndarray]  per detector, (N_asd, F)
            self.n_recolour_asd: int
        """
        self._recolour_banks = []
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

            bank = np.empty((n_asd, n_freq), dtype=np.float32)
            with open(bin_path, "rb") as fh:
                nread = fh.readinto(memoryview(bank).cast("B"))
            if nread != expected:
                raise ValueError(
                    f"recolour ASD bank {bin_path}: read {nread} of {expected} bytes"
                )
            self._recolour_banks.append(bank)

            if self.n_recolour_asd is None:
                self.n_recolour_asd = n_asd

    @torch.no_grad()
    def forward(
        self,
        batch_td: torch.Tensor,
        segment_ids: torch.Tensor,
        run_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        torch.compile-safe FD recolouring.

        ``run_ids`` (B, D) tags each window's source run so the whitening PSD is
        keyed by (run_id, segment_index). ``None`` means single-run (all run 0) —
        the case for single-run training and the hard-noise miner's reader.
        """

        # TD to FD (B, D, F)
        X = torch.fft.rfft(batch_td, dim=-1, norm="forward")

        # Actual batch size from the input — may differ from the configured
        # training batch ``self.B`` (e.g. the hard-noise miner reads variable
        # batches), so everything below is sized off ``B``, not ``self.B``.
        B = batch_td.shape[0]

        # Bernoulli recolour mask (B, D, 1)
        mask_cpu = torch.rand(B, self.D, 1) < self.p_recolour
        mask = mask_cpu.to(X.device, non_blocking=True)

        # Whitening PSD: each window's own segment ASD, keyed by (run, segment).
        seg_idx = segment_ids.detach().cpu().numpy()
        run_idx = (np.zeros_like(seg_idx) if run_ids is None
                   else run_ids.detach().cpu().numpy())
        gathered_seg_asd = self._gather_segment(seg_idx, run_idx)
        gathered_seg_asd = gathered_seg_asd.to(X.device, non_blocking=True)

        X = torch.where(
            mask,
            X / (gathered_seg_asd + self.eps),
            X,
        )

        # Recolour PSD: a random ASD from the target-epoch bank
        recol_idx = torch.randint(0, self.n_recolour_asd, (B, self.D)).numpy()
        gathered_recol_asd = self._gather(self._recolour_banks, recol_idx)
        gathered_recol_asd = gathered_recol_asd.to(X.device)

        recol_gain = gathered_recol_asd + self.eps

        X = torch.where(mask, X * recol_gain, X)

        return X

    def _gather(self, banks, idx):
        """Gather a (B, D, F) ASD tensor from the per-detector in-RAM banks.

        ``idx`` is an integer array of shape (B, D); row ``idx[b, d]`` is picked
        from detector ``d``'s bank. Fancy indexing on the resident arrays is a
        RAM-speed copy (the NFS-memmap variant cost ~600 ms/gather).
        """
        F = banks[0].shape[1]
        out = np.empty((idx.shape[0], self.D, F), dtype=np.float32)
        for d in range(self.D):
            out[:, d, :] = banks[d][idx[:, d]]
        return torch.from_numpy(out)

    def _gather_segment(self, seg_idx, run_idx):
        """Gather a (B, D, F) segment ASD keyed by (run_id, segment_index).

        ``seg_idx`` / ``run_idx`` are (B, D) integer arrays. For each (detector,
        run) it fancy-indexes that run's bank for the windows belonging to it, so
        the cost matches the single-run gather (one fancy-index per detector when
        ``n_runs == 1``).
        """
        F = self._segment_banks[0][0].shape[1]
        out = np.empty((seg_idx.shape[0], self.D, F), dtype=np.float32)
        n_runs = len(self._segment_banks)
        for d in range(self.D):
            if n_runs == 1:
                out[:, d, :] = self._segment_banks[0][d][seg_idx[:, d]]
                continue
            for r in range(n_runs):
                m = run_idx[:, d] == r
                if m.any():
                    out[m, d, :] = self._segment_banks[r][d][seg_idx[m, d]]
        return torch.from_numpy(out)
