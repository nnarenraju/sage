#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hard sample mining for gravitational-wave detection training.

Two complementary buffers:
  HardSampleBuffer  — generic CPU-resident store of pre-processed tensors +
                      targets + logit scores.  Replaced wholesale each mining
                      epoch.  Convention: HIGHER logit stored = HARDER example.
                        • noise buffer  : highest ranking-stat backgrounds
                          (false-alarm candidates the model struggles to reject)
                        • signal buffer : LOWEST ranking-stat signals passed as
                          -logit so the worst missed detections rank highest.

  HardSampleMiner   — orchestrates the mining pass; drives both buffers.
"""

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from contextlib import nullcontext

from sage.core.config import get_cfg


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

class HardSampleBuffer:
    """
    CPU-resident buffer of hard training examples.

    Replaced wholesale by HardSampleMiner.mine() every K epochs.
    All tensors are stored as float32 on CPU and moved to the target
    device on demand.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data    = None   # (K, ...) float32, CPU
        self._targets = None   # (K, T)   float32, CPU — None for noise buffer
        self._logits  = None   # (K,)     float32, CPU
        self.is_ready = False

    # ------------------------------------------------------------------
    def replace(
        self,
        data:    torch.Tensor,
        logits:  torch.Tensor,
        targets: torch.Tensor = None,
    ):
        """
        Keep the top-`capacity` hardest examples (highest logit).

        For the signal buffer call with logits = -ranking_stat so that the
        worst missed-detections (lowest real logit) rank first.
        """
        k = min(len(logits), self.capacity)
        _, topk_idx = torch.topk(logits.float().cpu(), k=k)

        self._data    = data[topk_idx].cpu().float()
        self._logits  = logits[topk_idx].cpu().float()
        self._targets = (
            targets[topk_idx].cpu().float() if targets is not None else None
        )
        self.is_ready = True

    # ------------------------------------------------------------------
    def sample(
        self,
        n:      int,
        device: str,
        dtype:  torch.dtype = torch.float32,
    ):
        """Return (data, targets|None) for n random items from the buffer."""
        if not self.is_ready or len(self._data) == 0:
            return None, None
        n   = min(n, len(self._data))
        idx = torch.randint(0, len(self._data), (n,))
        data    = self._data[idx].to(device=device, dtype=dtype, non_blocking=True)
        targets = (
            self._targets[idx].to(device=device, dtype=dtype, non_blocking=True)
            if self._targets is not None else None
        )
        return data, targets

    def __len__(self):
        return len(self._data) if self._data is not None else 0

    @property
    def top_logit(self):
        if not self.is_ready:
            return float("nan")
        return self._logits[0].item()

    @property
    def median_logit(self):
        if not self.is_ready:
            return float("nan")
        return self._logits.median().item()


# ---------------------------------------------------------------------------
# Miner
# ---------------------------------------------------------------------------

class HardSampleMiner:
    """
    Periodic mining pass that populates HardSampleBuffer instances.

    Noise mining  — find background windows the model ranks highest.
    Signal mining — find signal+noise windows the model ranks lowest
                    (missed detections).

    Both mining passes use the model in eval / inference mode;
    no gradient computation or weight update occurs.

    Parameters
    ----------
    hard_noise_buffer, hard_signal_buffer : HardSampleBuffer
    n_mine_noise : int
        Number of noise windows to evaluate per mining pass.
    n_mine_signal : int
        Number of signal+noise windows to evaluate per mining pass.
    mine_batch_size : int
        Batch size used during mining (independent of training batch size).
        Must be >= cfg.batch_size for signal mining (signal sampler is fixed).
    """

    def __init__(
        self,
        hard_noise_buffer:  HardSampleBuffer,
        hard_signal_buffer: HardSampleBuffer,
        n_mine_noise:   int = 100_000,
        n_mine_signal:  int =  50_000,
        mine_batch_size: int = 256,
    ):
        self.hard_noise_buffer  = hard_noise_buffer
        self.hard_signal_buffer = hard_signal_buffer
        self.n_mine_noise       = n_mine_noise
        self.n_mine_signal      = n_mine_signal
        self.mine_batch_size    = mine_batch_size

    # ------------------------------------------------------------------
    @torch.no_grad()
    def mine(
        self,
        model,
        noise_sampler,
        signal_sampler,
        processor,
        device:   str,
        autocast: bool = False,
    ):
        """
        Run full mining pass and update both buffers.

        Uses the configured batch size for signal mining (signal sampler is
        fixed at cfg.batch_size).  Uses mine_batch_size for noise mining.
        """
        cfg = get_cfg()
        sig_bs = cfg.batch_size   # signal sampler always uses this

        was_training = model.training
        model.eval()

        cast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if autocast else nullcontext()
        )

        # ----------------------------------------------------------------
        # Mine hard background
        # ----------------------------------------------------------------
        print("  [Miner] Mining hard background windows …")
        all_proc, all_logits = [], []
        n_batches = max(1, self.n_mine_noise // self.mine_batch_size)

        for _ in tqdm(range(n_batches), desc="  hard-bg", leave=False):
            noise_fd = noise_sampler._read_batch(self.mine_batch_size)
            x = processor(noise_fd)
            with cast:
                out = model(x)
            logits = out[0].reshape(-1).float().cpu()
            all_proc.append(x.float().cpu())
            all_logits.append(logits)

        noise_proc   = torch.cat(all_proc,   dim=0)
        noise_logits = torch.cat(all_logits, dim=0)
        self.hard_noise_buffer.replace(noise_proc, noise_logits)

        print(
            f"  [Miner] Noise buffer: {len(self.hard_noise_buffer):,} samples | "
            f"top logit={self.hard_noise_buffer.top_logit:.3f} | "
            f"median={self.hard_noise_buffer.median_logit:.3f}"
        )

        # ----------------------------------------------------------------
        # Mine hard signals (worst missed detections)
        # ----------------------------------------------------------------
        print("  [Miner] Mining hard signal windows …")
        all_sig_proc, all_sig_tgt, all_sig_logits = [], [], []
        n_batches_sig = max(1, self.n_mine_signal // sig_bs)

        for _ in tqdm(range(n_batches_sig), desc="  hard-sig", leave=False):
            signal_fd, signal_targets = signal_sampler()              # (B, D, F), (B, T)
            noise_fd = noise_sampler._read_batch(sig_bs)              # (B, D, F)
            # Inject all B signals (maximises mining coverage per batch)
            x = (noise_fd + signal_fd).detach()
            x = processor(x)
            with cast:
                out = model(x)
            logits = out[0].reshape(-1).float().cpu()

            all_sig_proc.append(x.float().cpu())
            all_sig_tgt.append(signal_targets.float().cpu())
            all_sig_logits.append(logits)

        sig_proc    = torch.cat(all_sig_proc,   dim=0)
        sig_targets = torch.cat(all_sig_tgt,    dim=0)
        sig_logits  = torch.cat(all_sig_logits, dim=0)

        # Negate so lowest ranking stat (hardest miss) → highest buffer score
        self.hard_signal_buffer.replace(sig_proc, -sig_logits, sig_targets)

        worst_logit  = sig_logits.min().item()
        median_logit = sig_logits.median().item()
        print(
            f"  [Miner] Signal buffer: {len(self.hard_signal_buffer):,} samples | "
            f"worst logit={worst_logit:.3f} | median={median_logit:.3f}"
        )

        if was_training:
            model.train()
