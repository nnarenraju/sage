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

    Memory
    ------
    The target recolour ASD bank is kept **resident in host RAM** (a per-batch
    random gather from the NFS mount costs ~600 ms and starves the GPU). It is
    LARGE: ``num_psds x n_freq x 4`` bytes ~= **16 GB per detector** (e.g. 250k
    ASDs x 16385 bins). So a 2-detector network needs ~33 GB just for recolour
    banks (~49 GB for 3 detectors), plus ~1.3 GB/detector of segment banks and
    ~15 GB torch/compile overhead. **Size the job's ``--mem`` accordingly**
    (>= ~96 GB for 2 detectors, ~128 GB for 3); requesting too little OOMs before
    training starts.

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
    forward(batch_td, segment_ids, run_ids=None) :
        ``batch_td``  : ``(B, D, T)`` float32 — time-domain noise windows.
        ``segment_ids``: ``(B, D)`` int64 — index into the per-run segment ASD
        bank (selects the per-segment whitening ASD).
        ``run_ids``    : ``(B, D)`` int64 or None — which pooled run each window
        came from (keys the per-run segment bank); ``None`` = single-run (run 0).

    Returns ``(B, D, F)`` complex64 — frequency-domain (recoloured) strain.
    """

    def __init__(
        self,
        *,
        p_recolour: float,
        recolour_dataset_dir: str,
        eps: float = 1e-38,
        seed: int | None = None,
        dr_gain: float = 0.5,
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

        # Dedicated CPU generator so the recolour augmentation draws (the
        # per-sample mask + the random target-ASD pick) are reproducible and
        # resume-aware, isolated from the main thread's global-RNG draws (this
        # runs inside the noise-sampler prefetch thread). seed=None -> the global
        # RNG (legacy, unseeded).
        self._gen = None
        if seed is not None:
            self._gen = torch.Generator()
            self._gen.manual_seed(int(seed))

        self.B = cfg.batch_size
        self.D = len(self.detectors)

        # We expect this length from the PSDs
        # Interpolate them after production
        self.n_freq = self.seq_len // 2 + 1

        # Domain randomization (Tobin et al., IROS 2017), DATA-DRIVEN bound: a
        # smooth, per-(sample, detector), frequency-dependent MULTIPLICATIVE gain
        # jitter on the recoloured target ASD, to fill the gaps between the
        # DISCRETE real target-run ASDs so the net generalises around that
        # manifold rather than only to its sampled points.
        #
        # The per-frequency jitter is capped at ``k * sigma(f)``, where
        # ``sigma(f)`` is the EMPIRICAL fractional spread (std/mean) of the
        # resident target ASD bank at that frequency and ``k = dr_gain`` in
        # [0, 1]. This makes the perturbation an interpolation strictly WITHIN the
        # real-ASD manifold (physical by construction, never extrapolation), and
        # it is naturally low-frequency weighted because the real ASDs vary most
        # there (measured sigma(f): ~0.51 at 15-30 Hz falling to ~0.18 above
        # 300 Hz). Because the gain is MULTIPLICATIVE it is a constant fractional
        # / dB perturbation -- correct in the log-ASD space where the real
        # variation lives. ``k = 0.5`` -> up to half the real per-frequency
        # spread. 0 disables. See notebooks/recolour_augmentation.ipynb.
        self.dr_gain = float(dr_gain)
        # Smooth Legendre-like shape basis (linear + quadratic tilt), precomputed
        # as compile-friendly constants. Combined in forward with random O(1)
        # coeffs and normalised so |shape| <= 1, so k*sigma(f) is a true per-bin
        # bound on the fractional gain deviation.
        _t = torch.linspace(-1.0, 1.0, self.n_freq).view(1, 1, -1)
        self._tnorm = _t                                   # (1, 1, F) linear
        self._tquad = _t * _t - 1.0 / 3.0                  # (1, 1, F) quadratic

        # Load PSDs to torch.float32 on GPU
        self._load_segment_asds()
        self._load_recolour_asds()

        # Data-driven per-frequency jitter bound sigma(f) (float64; see method).
        self._dr_sigma = None
        if self.dr_gain > 0.0:
            self._compute_dr_sigma()

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

    def _compute_dr_sigma(self):
        """
        Data-driven domain-randomization bound ``sigma(f) = std/mean`` of the
        resident recolour ASD bank, per detector, on the training frequency grid.

        Computed in FLOAT64: ASD values are ~1e-24, so ASD^2 ~1e-48 underflows
        float32 (min-normal ~1.2e-38) and the variance silently reads as 0. The
        bank is streamed in chunks so the float64 upcast never materialises the
        whole ~16 GB/detector array at once. Result is a (D, F) float32 CPU
        tensor consumed multiplicatively in :meth:`forward`.
        """
        sig = np.empty((self.D, self.n_freq), dtype=np.float32)
        for d, bank in enumerate(self._recolour_banks):
            n = bank.shape[0]
            s1 = np.zeros(self.n_freq, dtype=np.float64)
            s2 = np.zeros(self.n_freq, dtype=np.float64)
            for i in range(0, n, 4096):
                chunk = np.asarray(bank[i:i + 4096], dtype=np.float64)
                s1 += chunk.sum(axis=0)
                s2 += (chunk * chunk).sum(axis=0)
            mu = s1 / n
            var = np.maximum(s2 / n - mu * mu, 0.0)
            sig[d] = (np.sqrt(var) / (mu + 1e-300)).astype(np.float32)
        self._dr_sigma = torch.from_numpy(sig)          # (D, F) CPU float32

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
        mask_cpu = torch.rand(B, self.D, 1, generator=self._gen) < self.p_recolour
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
        recol_idx = torch.randint(
            0, self.n_recolour_asd, (B, self.D), generator=self._gen
        ).numpy()
        gathered_recol_asd = self._gather(self._recolour_banks, recol_idx)  # (B,D,F) CPU

        # Data-driven domain-randomization gain jitter (Tobin et al. IROS 2017):
        # a smooth broadband + linear + quadratic frequency shape, per (sample,
        # detector), scaled per frequency by the data-driven bound k*sigma(f).
        # Multiplicative -> constant fractional / dB perturbation (log-scale
        # correct); per-frequency -> naturally low-frequency weighted. Drawn from
        # the same seeded generator as the mask (reproducible / resume-aware).
        if self.dr_gain > 0.0:
            k = self.dr_gain
            g = self._gen
            # Random O(1) shape coeffs per (sample, detector), normalised so the
            # smooth shape lies in ~[-1, 1] (max |b0| + |b1*t| + |b2*(t^2-1/3)|
            # = 1 + 1 + 2/3 = 8/3), making k*sigma(f) an exact per-bin bound.
            b0 = 2.0 * torch.rand(B, self.D, 1, generator=g) - 1.0
            b1 = 2.0 * torch.rand(B, self.D, 1, generator=g) - 1.0
            b2 = 2.0 * torch.rand(B, self.D, 1, generator=g) - 1.0
            shape = (b0 + b1 * self._tnorm + b2 * self._tquad) * (3.0 / 8.0)
            # k*sigma(f) per-frequency gain; clamp is a final physical safety rail.
            gain = (1.0 + k * self._dr_sigma.unsqueeze(0) * shape).clamp(0.3, 3.0)
            gathered_recol_asd = gathered_recol_asd * gain

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
