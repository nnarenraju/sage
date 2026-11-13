#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Noise sampler for offline testing that records where each window came
from, so a window can be re-read later
(:func:`sage.factory.testing.reconstruct_noise`).
"""

import numpy as np
import torch

from .real_noise import MemmapNoiseSampler
from ._pycbc_lazy import dyn_range_fac


class TestNoiseSampler(MemmapNoiseSampler):
    """
    ``MemmapNoiseSampler`` that records each batch's provenance.

    Batches are random windows without hard-window replay or
    postprocess; ``last_provenance`` holds the per-detector
    ``(run, segment, start)`` of the last batch popped.
    """

    def __init__(self, *args, **kwargs):
        # Set before the prefetch thread starts in super().__init__
        self.last_provenance = None
        super().__init__(*args, **kwargs)

    def _read_batch(self, batch_size):
        """
        Read one batch of random windows and its provenance.

        Parameters
        ----------
        batch_size : int
            Number of windows.

        Returns
        -------
        batch : torch.Tensor, shape ``(B, D, F)``
            ``rfft`` of the windows.
        prov : dict
            ``"start"``, ``"segment"`` and ``"run"``, each ``(B, D)``
            int64.
        """
        B, D, seq_len = batch_size, self.n_detectors, self.seq_len
        start_indices, segment_indices, run_indices = self._sample_starts_batch(B)

        batch_tensor = torch.empty((B, D, seq_len), dtype=torch.float32,
                                   device=self.device)
        arrs = [np.empty((B, seq_len), dtype=np.float32) for _ in range(D)]
        mmaps = self.mmaps

        def _read_window(job):
            """Read window ``i`` of detector ``d`` into ``arrs``."""
            d, i = job
            s = int(start_indices[d][i]); r = int(run_indices[d][i])
            arrs[d][i] = mmaps[d][r][s:s + seq_len]

        list(self._read_pool.map(
            _read_window, [(d, i) for d in range(D) for i in range(B)]))

        pin = torch.device(self.device).type == "cuda"
        prov = {"start":   np.empty((B, D), dtype=np.int64),
                "segment": np.empty((B, D), dtype=np.int64),
                "run":     np.empty((B, D), dtype=np.int64)}
        for d in range(D):
            arrs[d] /= dyn_range_fac()                 # undo the scaling
            cpu_tensor = torch.from_numpy(arrs[d])
            if pin:
                cpu_tensor = cpu_tensor.pin_memory()
            batch_tensor[:, d, :].copy_(cpu_tensor, non_blocking=pin)
            prov["start"][:, d]   = np.asarray(start_indices[d], dtype=np.int64)
            prov["segment"][:, d] = np.asarray(segment_indices[d], dtype=np.int64)
            prov["run"][:, d]     = np.asarray(run_indices[d], dtype=np.int64)

        batch_tensor = torch.fft.rfft(batch_tensor, dim=-1, norm="forward")
        return batch_tensor, prov

    def sample_batch(self):
        """
        Pop a batch from the prefetch queue and keep its provenance.

        Returns
        -------
        torch.Tensor
            FD noise batch.
        """
        batch_tensor, prov = self.queue.get()
        self.last_provenance = prov
        return batch_tensor
