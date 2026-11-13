#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Recolouring of noise windows in the frequency domain: whitening by each
window's own segment ASD, then colouring by a random target-run ASD.

Filename        : recolour.py
Description     : Short description of the file

Created on 2026-02-09 23:39:57

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

import os
import json
import torch
import numpy as np

from pathlib import Path

from sage.config.machine import recolour_bank_file, segment_asd_file, segments_file
from sage.core.geometry import Window
from sage.core.interpolation import torch_linear_interp


# Recolour bank rows a ``sage check`` build reads per detector
CHECK_BANK_ROWS = 128


def load_segment_asds(data_dir, detectors, mapped=False):
    """
    Read a release's per-segment ASD banks (``segment_psds/``).

    Parameters
    ----------
    data_dir : str or Path
        The release's data directory.
    detectors : list of str
        Detectors, in channel order.
    mapped : bool
        Memory-map the files instead of reading them into memory.

    Returns
    -------
    banks : list of numpy.ndarray
        One ``(N_seg, F)`` float32 array per detector, in sidecar order.
    n_freq : int
        ``F``, shared by every detector.

    Raises
    ------
    ValueError
        If the detectors' grids differ, or a file's size or read length
        does not match its sidecar.
    """
    banks = []
    n_freq = None
    for det in detectors:
        bin_path = segment_asd_file(data_dir, det)
        meta_path = segments_file(bin_path)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        n_seg = len(meta)
        det_n_freq = int(meta[0]["psd_len"])  # one F per file
        if n_freq is None:
            n_freq = det_n_freq
        elif det_n_freq != n_freq:
            raise ValueError(
                f"segment ASD bank {bin_path}: psd_len {det_n_freq} != {n_freq} "
                f"of the other detectors; one release must share one grid"
            )
        expected = n_seg * det_n_freq * 4
        actual = os.path.getsize(bin_path)
        if actual != expected:
            raise ValueError(
                f"segment ASD bank {bin_path}: size {actual} != expected "
                f"{expected} ({n_seg} x {det_n_freq} float32) — non-uniform psd_len?"
            )

        if mapped:
            banks.append(np.memmap(bin_path, dtype=np.float32, mode="r", shape=(n_seg, det_n_freq)))
            continue
        # Read into a preallocated buffer, without an intermediate copy
        bank = np.empty((n_seg, det_n_freq), dtype=np.float32)
        with open(bin_path, "rb") as fh:
            nread = fh.readinto(memoryview(bank).cast("B"))
        if nread != expected:
            raise ValueError(
                f"segment ASD bank {bin_path}: read {nread} of {expected} bytes"
            )
        banks.append(bank)
    return banks, n_freq


class RecolourPostprocess(torch.nn.Module):
    """
    Random recolouring of noise windows in the frequency domain.

    With ``probability`` per window and detector, the window is
    whitened by its own segment ASD and coloured by a random ASD of the
    target run's recolour bank, optionally with a smooth gain jitter.
    Banks are read into memory; ASDs on another grid are interpolated.

    Parameters
    ----------
    runs : list of dict
        The training sampler's runs, ``{"run", "data_dir", "bins"}``, in
        its order; ``data_dir`` holds each run's per-segment ASDs.
    target_asd_dir : Path
        ASD directory of the target observing run; its
        ``recolour_psds/`` holds the per-detector recolour ASD banks.
    probability : float
        Probability in ``[0, 1]`` of recolouring a window.
    asd_jitter_gain : float
        Bound in ``[0, 1]`` of the gain jitter on the target ASD, as a
        fraction of the bank's per-frequency spread; 0 disables it.
    seed : int
        Seed of the recolour draws.
    detectors : list of str
        Detectors, in channel order.
    window : Window
        Window geometry; sets the frequency grid.
    device : str or torch.device
        Device of the output.
    small : bool
        For ``sage check``: read ``CHECK_BANK_ROWS`` recolour rows and
        memory-map the segment ASDs.

    Raises
    ------
    ValueError
        If the banks' grids differ between runs or detectors, or a bank
        file's size does not match its sidecar.
    """

    def __init__(
        self,
        *,
        runs: list,
        target_asd_dir: Path,
        probability: float,
        asd_jitter_gain: float,
        seed: int,
        detectors: list,
        window: Window,
        device,
        small: bool = False,
    ):
        super().__init__()
        self.small = small

        # In the sampler's order, so banks line up with its run ids
        self._runs = runs

        self.target_asd_dir = Path(target_asd_dir)
        self.detectors = list(detectors)
        self.seq_len = window.padded_length_in_samples
        self.sample_rate = window.sample_rate_in_hz
        self.probability = float(probability)
        self.device = device
        # Added to ASDs before dividing or multiplying, for quiet bins.
        self.eps = 1e-38

        # Own CPU generator: the draws are reproducible and independent
        # of the global RNG (this runs in the prefetch thread)
        self._gen = torch.Generator()
        self._gen.manual_seed(int(seed))

        self.D = len(self.detectors)

        # Signal grid: rFFT of the padded window over [0, Nyquist];
        # banks on another grid are interpolated onto it
        self.n_freq = self.seq_len // 2 + 1
        self._sig_freqs = torch.linspace(
            0.0, self.sample_rate / 2.0, self.n_freq, dtype=torch.float64
        )

        # Domain randomisation: a smooth
        # multiplicative gain on the target ASD per window and detector,
        # bounded per frequency by asd_jitter_gain * sigma(f), sigma(f)
        # being the std/mean of the recolour bank at f
        self.asd_jitter_gain = float(asd_jitter_gain)
        # Linear and quadratic shape basis of the gain jitter
        _t = torch.linspace(-1.0, 1.0, self.n_freq).view(1, 1, -1)
        self._tnorm = _t                                   # (1, 1, F) linear
        self._tquad = _t * _t - 1.0 / 3.0                  # (1, 1, F) quadratic

        self._load_segment_asds()
        self._load_recolour_asds()

        self._dr_sigma = None
        if self.asd_jitter_gain > 0.0:
            self._compute_dr_sigma()

    def _load_segment_asds(self):
        """
        Load the per-segment ASD banks of every run and detector.

        Sets ``_segment_banks[run_id][d]``, an ``(N_seg, F)`` array
        indexed by segment index.

        Raises
        ------
        ValueError
            If the runs' segment ASDs are on different grids.
        """
        self._segment_banks = []
        seg_n_freq = None   # shared by all pooled runs

        for run in self._runs:
            det_banks, n_freq = load_segment_asds(run["data_dir"], self.detectors, self.small)
            # Runs are gathered into one buffer on the first run's grid
            if seg_n_freq is None:
                seg_n_freq = n_freq
            elif n_freq != seg_n_freq:
                raise ValueError(
                    f"segment ASD banks under {run['data_dir']}: psd_len {n_freq} != "
                    f"{seg_n_freq} from an earlier pooled run; all runs' segment ASDs "
                    f"must share one frequency grid to be pooled for recolour."
                )
            self._segment_banks.append(det_banks)

        # Bank grid; None if it is the signal grid
        self._seg_src_freqs = self._bank_src_freqs(seg_n_freq)

    def _load_recolour_asds(self):
        """
        Load the recolour ASD bank of every detector into memory.

        Each detector is read into its own preallocated buffer. Sets
        ``_recolour_banks`` (``(N_asd, F)`` per detector) and
        ``n_recolour_asd``, the first detector's row count.

        Raises
        ------
        ValueError
            If a file's size or read length does not match its sidecar.
        """
        self._recolour_banks = []
        self.n_recolour_asd = None

        for det in self.detectors:
            bin_path = recolour_bank_file(self.target_asd_dir, det)
            meta_path = bin_path.with_suffix(".json")

            with open(meta_path, "r") as f:
                meta = json.load(f)

            n_asd = int(meta["num_psds"])
            n_freq = int(meta["num_freq_bins"])
            expected = n_asd * n_freq * 4
            actual = os.path.getsize(bin_path)
            if actual != expected:
                raise ValueError(
                    f"recolour ASD bank {bin_path}: size {actual} != expected "
                    f"{expected} ({n_asd} x {n_freq} float32)"
                )

            if self.small:
                n_asd = min(n_asd, CHECK_BANK_ROWS)
                expected = n_asd * n_freq * 4
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

        # Bank grid; None if it is the signal grid
        self._recol_src_freqs = self._bank_src_freqs(n_freq)

    def _compute_dr_sigma(self):
        """
        Set ``_dr_sigma``, the std/mean of each recolour bank per bin.

        Accumulated in float64 over chunks of rows, as squared ASDs
        underflow float32; stored as a ``(D, F)`` float32 CPU tensor on
        the signal grid.
        """
        nfb = self._recolour_banks[0].shape[1]          # bank grid
        sig = np.empty((self.D, nfb), dtype=np.float32)
        for d, bank in enumerate(self._recolour_banks):
            n = bank.shape[0]
            s1 = np.zeros(nfb, dtype=np.float64)
            s2 = np.zeros(nfb, dtype=np.float64)
            for i in range(0, n, 4096):
                chunk = np.asarray(bank[i:i + 4096], dtype=np.float64)
                s1 += chunk.sum(axis=0)
                s2 += (chunk * chunk).sum(axis=0)
            mu = s1 / n
            var = np.maximum(s2 / n - mu * mu, 0.0)
            sig[d] = (np.sqrt(var) / (mu + 1e-300)).astype(np.float32)
        # Onto the signal grid, like the recolour ASDs
        self._dr_sigma = self._to_signal_grid(
            torch.from_numpy(sig), self._recol_src_freqs
        )                                               # (D, n_freq)

    @torch.no_grad()
    def forward(
        self,
        batch_td: torch.Tensor,
        segment_ids: torch.Tensor,
        run_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Convert windows to FD and recolour each with ``probability``.

        Parameters
        ----------
        batch_td : torch.Tensor, shape ``(B, D, T)``
            Time-domain noise windows.
        segment_ids : torch.Tensor, shape ``(B, D)``
            Segment index of each window, selecting its own ASD.
        run_ids : torch.Tensor, shape ``(B, D)``, or None
            Pooled run of each window; ``None`` is run 0.

        Returns
        -------
        torch.Tensor, shape ``(B, D, F)``
            ``rfft`` of the windows (``norm="forward"``), recoloured
            where drawn.
        """
        X = torch.fft.rfft(batch_td, dim=-1, norm="forward")

        # Batch sizes vary, e.g. in the hard-noise miner
        B = batch_td.shape[0]

        mask, seg_asd, target_asd = self._draw(
            B, segment_ids, run_ids, self._gen, X.device
        )

        # Whiten by the segment ASD, colour by the target ASD
        X = torch.where(mask, X / (seg_asd + self.eps), X)
        X = torch.where(mask, X * (target_asd + self.eps), X)

        return X

    @torch.no_grad()
    def spectra(self, segment_ids, run_ids=None, generator=None):
        """
        ASD each window carries after this postprocess.

        A window left alone keeps its segment ASD, a recoloured one the
        jittered target ASD. Drawn as :meth:`forward` draws, but from
        ``generator``, so the training draws are unchanged.

        Parameters
        ----------
        segment_ids : torch.Tensor, shape ``(B, D)``
            Segment index of each window.
        run_ids : torch.Tensor, shape ``(B, D)``, or None
            Pooled run of each window; ``None`` is run 0.
        generator : torch.Generator or None
            Generator of the draws.

        Returns
        -------
        torch.Tensor, shape ``(B, D, F)``
            ASDs on CPU.
        """
        mask, seg_asd, target_asd = self._draw(
            segment_ids.shape[0], segment_ids, run_ids, generator, "cpu"
        )
        return torch.where(mask, target_asd, seg_asd)

    @torch.no_grad()
    def target_segment_spectra(self, segment_ids):
        """
        Segment ASDs of target-run windows read without recolouring.

        The target run's segment banks are loaded on the first call.

        Parameters
        ----------
        segment_ids : torch.Tensor, shape ``(B, D)``
            Indices into the target run's segment sidecar.

        Returns
        -------
        torch.Tensor, shape ``(B, D, F)``
            ASDs on the signal grid.
        """
        if getattr(self, "_target_segment_banks", None) is None:
            banks, n_freq = load_segment_asds(self.target_asd_dir, self.detectors, self.small)
            self._target_segment_banks = banks
            self._target_seg_src_freqs = self._bank_src_freqs(n_freq)
        asd = self._gather(
            self._target_segment_banks, segment_ids.detach().cpu().numpy()
        )
        return self._to_signal_grid(asd, self._target_seg_src_freqs)

    def _draw(self, B, segment_ids, run_ids, generator, device):
        """
        The random draws of one recolour pass.

        Parameters
        ----------
        B : int
            Batch size.
        segment_ids : torch.Tensor, shape ``(B, D)``
            Segment index of each window.
        run_ids : torch.Tensor, shape ``(B, D)``, or None
            Pooled run of each window; ``None`` is run 0.
        generator : torch.Generator or None
            Generator of the draws.
        device : str or torch.device
            Device of the outputs.

        Returns
        -------
        mask : torch.Tensor, shape ``(B, D, 1)``
            Whether each window is recoloured.
        seg_asd, target_asd : torch.Tensor, shape ``(B, D, F)``
            Segment ASD and jittered target ASD of each window.
        """
        mask_cpu = torch.rand(B, self.D, 1, generator=generator) < self.probability
        mask = mask_cpu.to(device, non_blocking=True)

        # Each window's own segment ASD, keyed by (run, segment)
        seg_idx = segment_ids.detach().cpu().numpy()
        run_idx = (np.zeros_like(seg_idx) if run_ids is None
                   else run_ids.detach().cpu().numpy())
        seg_asd = self._gather_segment(seg_idx, run_idx)
        seg_asd = seg_asd.to(device, non_blocking=True)
        seg_asd = self._to_signal_grid(seg_asd, self._seg_src_freqs)

        # A random ASD of the target run's recolour bank
        recol_idx = torch.randint(
            0, self.n_recolour_asd, (B, self.D), generator=generator
        ).numpy()
        target_asd = self._gather(self._recolour_banks, recol_idx)  # bank grid
        target_asd = self._to_signal_grid(target_asd, self._recol_src_freqs)

        # Gain jitter: constant + linear + quadratic shape per window
        # and detector, scaled per frequency by k * sigma(f)
        if self.asd_jitter_gain > 0.0:
            k = self.asd_jitter_gain
            g = generator
            # Coefficients in [-1, 1]; 3/8 keeps |shape| <= 1, as
            # max |b0| + |b1 t| + |b2 (t^2 - 1/3)| = 1 + 1 + 2/3 = 8/3
            b0 = 2.0 * torch.rand(B, self.D, 1, generator=g) - 1.0
            b1 = 2.0 * torch.rand(B, self.D, 1, generator=g) - 1.0
            b2 = 2.0 * torch.rand(B, self.D, 1, generator=g) - 1.0
            shape = (b0 + b1 * self._tnorm + b2 * self._tquad) * (3.0 / 8.0)
            # Gain clamped to [0.3, 3]
            gain = (1.0 + k * self._dr_sigma.unsqueeze(0) * shape).clamp(0.3, 3.0)
            target_asd = target_asd * gain

        return mask, seg_asd, target_asd.to(device)

    def _gather(self, banks, idx):
        """
        Gather row ``idx[b, d]`` of bank ``d`` for each window.

        Parameters
        ----------
        banks : list of numpy.ndarray
            Per-detector ``(N, F)`` banks.
        idx : numpy.ndarray, shape ``(B, D)``
            Row indices.

        Returns
        -------
        torch.Tensor, shape ``(B, D, F)``
            Gathered ASDs, float32 on CPU.
        """
        F = banks[0].shape[1]
        out = np.empty((idx.shape[0], self.D, F), dtype=np.float32)
        for d in range(self.D):
            out[:, d, :] = banks[d][idx[:, d]]
        return torch.from_numpy(out)

    def _gather_segment(self, seg_idx, run_idx):
        """
        Gather each window's segment ASD, keyed by (run, segment).

        Parameters
        ----------
        seg_idx, run_idx : numpy.ndarray, shape ``(B, D)``
            Segment indices and run ids.

        Returns
        -------
        torch.Tensor, shape ``(B, D, F)``
            Segment ASDs on the bank grid, float32 on CPU.
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

    # A bank is built on its run's padded grid; banks on another grid
    # have only the sampled ASDs interpolated, per batch

    def _bank_src_freqs(self, src_n_freq):
        """
        Frequency axis of a bank with ``src_n_freq`` bins.

        Parameters
        ----------
        src_n_freq : int
            Bins of the bank over ``[0, Nyquist]``.

        Returns
        -------
        torch.Tensor or None
            The axis, float64, or ``None`` if the bank is on the signal
            grid.
        """
        if int(src_n_freq) == int(self.n_freq):
            return None
        return torch.linspace(
            0.0, self.sample_rate / 2.0, int(src_n_freq), dtype=torch.float64
        )

    def _to_signal_grid(self, asd, src_freqs):
        """
        Interpolate ASDs linearly onto the signal grid.

        Parameters
        ----------
        asd : torch.Tensor, shape ``(..., F_src)``
            ASDs on the bank grid.
        src_freqs : torch.Tensor or None
            Bank frequency axis; ``None`` returns ``asd`` unchanged.

        Returns
        -------
        torch.Tensor, shape ``(..., n_freq)``
            ASDs on the signal grid, in the dtype of ``asd``.
        """
        if src_freqs is None:
            return asd
        out = torch_linear_interp(
            self._sig_freqs.to(asd.device),
            src_freqs.to(asd.device),
            asd,
        )
        return out.to(asd.dtype)
