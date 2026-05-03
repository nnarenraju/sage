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
        """Highest stored logit, or NaN if the buffer is empty."""
        if not self.is_ready:
            return float("nan")
        return self._logits[0].item()

    @property
    def median_logit(self):
        """Median of all stored logits, or NaN if the buffer is empty."""
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

    Thread-safety note
    ------------------
    MemmapNoiseSampler runs a prefetch thread that also drives _read_batch /
    _sample_starts_batch, which share a non-thread-safe numpy RNG.  Calling
    _read_batch from the main thread while the prefetch thread is running
    causes a data race.  We avoid this entirely by consuming pre-fetched
    batches through sample_batch() — which only touches the thread-safe Queue
    — rather than calling _read_batch directly.  The batch size for mining
    therefore matches cfg.batch_size; n_mine_noise / n_mine_signal are rounded
    up to the nearest multiple automatically.

    Memory note
    -----------
    Instead of accumulating all processed tensors in RAM and selecting top-K
    at the end (O(n_mine × tensor_size) memory), we prune to top-K every
    `prune_every` batches (O(prune_every × batch_size + K) memory).

    Parameters
    ----------
    hard_noise_buffer, hard_signal_buffer : HardSampleBuffer
    n_mine_noise : int
        Approximate number of noise windows to evaluate per mining pass.
    n_mine_signal : int
        Approximate number of signal+noise windows to evaluate per mining pass.
    prune_every : int
        Prune accumulated candidates to buffer capacity every this many batches.
    """

    def __init__(
        self,
        hard_noise_buffer:  HardSampleBuffer,
        hard_signal_buffer: HardSampleBuffer,
        n_mine_noise:   int = 100_000,
        n_mine_signal:  int =  50_000,
        prune_every:    int = 20,
    ):
        self.hard_noise_buffer  = hard_noise_buffer
        self.hard_signal_buffer = hard_signal_buffer
        self.n_mine_noise       = n_mine_noise
        self.n_mine_signal      = n_mine_signal
        self.prune_every        = prune_every

    # ------------------------------------------------------------------
    @staticmethod
    def _streaming_top_k(
        new_data:    torch.Tensor,
        new_logits:  torch.Tensor,
        new_targets,
        acc_data:    list,
        acc_logits:  list,
        acc_targets: list,
        capacity:    int,
    ):
        """
        Append new batch to accumulator lists, then prune to top-`capacity`
        to keep memory bounded.
        """
        acc_data.append(new_data.cpu().float())
        acc_logits.append(new_logits.cpu().float())
        if new_targets is not None:
            acc_targets.append(new_targets.cpu().float())

        # Prune to capacity
        all_l = torch.cat(acc_logits, dim=0)
        if len(all_l) > capacity:
            k = min(capacity, len(all_l))
            _, topk_idx = torch.topk(all_l, k=k)
            all_d = torch.cat(acc_data, dim=0)[topk_idx]
            all_l = all_l[topk_idx]
            acc_data.clear()
            acc_logits.clear()
            acc_data.append(all_d)
            acc_logits.append(all_l)
            if acc_targets:
                all_t = torch.cat(acc_targets, dim=0)[topk_idx]
                acc_targets.clear()
                acc_targets.append(all_t)

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

        Uses noise_sampler.sample_batch() (thread-safe Queue consumer) so the
        prefetch thread can remain running throughout.  The effective batch size
        is therefore cfg.batch_size; n_mine values are rounded up accordingly.
        """
        cfg    = get_cfg()
        bs     = cfg.batch_size
        cap_n  = self.hard_noise_buffer.capacity
        cap_s  = self.hard_signal_buffer.capacity

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
        acc_data, acc_logits = [], []
        n_batches = max(1, -(-self.n_mine_noise // bs))  # ceiling division

        for i in tqdm(range(n_batches), desc="  hard-bg", leave=False):
            # sample_batch() pulls from the prefetch queue — thread-safe
            noise_fd = noise_sampler.sample_batch()
            x        = processor(noise_fd)
            with cast:
                out = model(x)
            logits = out[0].reshape(-1).float().cpu()

            if (i + 1) % self.prune_every == 0 or i == n_batches - 1:
                self._streaming_top_k(
                    x, logits, None,
                    acc_data, acc_logits, [],
                    cap_n,
                )
            else:
                acc_data.append(x.float().cpu())
                acc_logits.append(logits)

        noise_proc   = torch.cat(acc_data,   dim=0)
        noise_logits = torch.cat(acc_logits, dim=0)
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
        acc_sig_data, acc_sig_logits, acc_sig_tgt = [], [], []

        # signal_sampler returns S = batch_size * class_balance items per call.
        # Use S (not bs) to count batches so n_mine_signal reflects actual
        # signal evaluations, not noise-batch evaluations.
        S = int(bs * cfg.class_balance)
        n_batches_sig = max(1, -(-self.n_mine_signal // S))

        for i in tqdm(range(n_batches_sig), desc="  hard-sig", leave=False):
            signal_fd, signal_targets = signal_sampler()
            # Pull a full noise batch from the prefetch queue (thread-safe)
            noise_fd = noise_sampler.sample_batch()

            # Mirror training: inject signals into random positions of the
            # noise batch so processor sees realistic signal+noise windows.
            # signal_fd.shape[0] == S; noise_fd.shape[0] == bs.
            sig_idx    = torch.randperm(bs)[:S]
            signal_pad = torch.zeros_like(noise_fd)
            signal_pad[sig_idx] = signal_fd

            x = processor(noise_fd + signal_pad)
            with cast:
                out = model(x)
            logits = out[0].reshape(-1).float().cpu()

            # Extract only the signal positions: these are the missed-detection
            # candidates.  sig_x: (S, ...), sig_logits: (S,), targets: (S, T).
            sig_x       = x[sig_idx]
            sig_logits  = logits[sig_idx.cpu()]
            sig_targets = signal_targets

            if (i + 1) % self.prune_every == 0 or i == n_batches_sig - 1:
                # Negate now so pruning keeps the LOWEST real logits (hardest misses)
                self._streaming_top_k(
                    sig_x, -sig_logits, sig_targets,
                    acc_sig_data, acc_sig_logits, acc_sig_tgt,
                    cap_s,
                )
            else:
                acc_sig_data.append(sig_x.float().cpu())
                acc_sig_logits.append(-sig_logits)      # negated: lowest → highest
                acc_sig_tgt.append(sig_targets.float().cpu())

        sig_proc    = torch.cat(acc_sig_data,   dim=0)
        sig_logits  = torch.cat(acc_sig_logits, dim=0)  # already negated
        sig_targets = torch.cat(acc_sig_tgt,    dim=0)
        self.hard_signal_buffer.replace(sig_proc, sig_logits, sig_targets)

        worst_real_logit = (-sig_logits).max().item()   # un-negate for display
        print(
            f"  [Miner] Signal buffer: {len(self.hard_signal_buffer):,} samples | "
            f"worst logit={worst_real_logit:.3f} | "
            f"median={(-sig_logits).median().item():.3f}"
        )

        if was_training:
            model.train()
