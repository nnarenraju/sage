#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testing-only noise sampler.

``TestNoiseSampler`` samples noise windows exactly like ``MemmapNoiseSampler`` but
also exposes each window's PROVENANCE -- per-detector ``(run, segment, start)`` --
via ``last_provenance``, so a flagged window can later be re-read verbatim from the
memmaps (see ``sage.factory.testing.reconstruct_noise``).

It SUBCLASSES ``MemmapNoiseSampler`` and overrides only the batch read + queue pop,
so the base sampler used by training/validation is left completely untouched. It
always draws the true background: no hard-noise biasing and no recolour
postprocess (both irrelevant to a sensitivity/FAR measurement).
"""

import numpy as np
import torch

from .real_noise import MemmapNoiseSampler
from ._pycbc_lazy import dyn_range_fac


class TestNoiseSampler(MemmapNoiseSampler):
    """Provenance-exposing noise sampler for offline testing (see module docstring)."""

    def __init__(self, *args, **kwargs):
        # Set before super().__init__ so it exists before the prefetch thread runs.
        self.last_provenance = None
        super().__init__(*args, **kwargs)

    def _read_batch(self, batch_size):
        """Read one batch AND its provenance -> ``(FD tensor (B,D,F), prov dict)``.

        Mirrors ``MemmapNoiseSampler._read_batch`` for the plain-background path
        (no hard-noise bias, no postprocess), and additionally returns the
        per-detector ``(run, segment, start)`` used for each window.
        """
        B, D, seq_len = batch_size, self.n_detectors, self.seq_len
        start_indices, segment_indices, run_indices = self._sample_starts_batch(B)

        batch_tensor = torch.empty((B, D, seq_len), dtype=torch.float32,
                                   device=self.device)
        arrs = [np.empty((B, seq_len), dtype=np.float32) for _ in range(D)]
        mmaps = self.mmaps

        def _read_window(job):
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
            arrs[d] /= dyn_range_fac()                 # restore original scale
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
        """Pop a batch from the prefetch queue and stash its provenance."""
        batch_tensor, prov = self.queue.get()
        self.last_provenance = prov
        return batch_tensor
