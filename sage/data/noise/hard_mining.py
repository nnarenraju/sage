#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hard noise mining for gravitational-wave detection training.

HardSampleBuffer  — CPU-resident store of pre-processed noise tensors.
                    Replaced wholesale each mining epoch.  Stores the
                    background windows that the model currently ranks
                    highest (false-alarm candidates it struggles to reject).

HardSampleMiner   — orchestrates the mining pass for background windows.
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
    CPU-resident buffer of hard background training examples.

    Replaced wholesale by HardSampleMiner.mine() every K epochs.
    All tensors are stored as float32 on CPU and moved to the target
    device on demand.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data    = None   # (K, ...) float32, CPU
        self._logits  = None   # (K,)     float32, CPU
        self.is_ready = False

    def replace(self, data: torch.Tensor, logits: torch.Tensor):
        """Keep the top-`capacity` hardest examples (highest logit)."""
        k = min(len(logits), self.capacity)
        _, topk_idx = torch.topk(logits.float().cpu(), k=k)
        self._data   = data[topk_idx].cpu().float()
        self._logits = logits[topk_idx].cpu().float()
        self.is_ready = True

    def sample(self, n: int, device: str, dtype: torch.dtype = torch.float32):
        """Return `n` random items from the buffer as a tensor on `device`."""
        if not self.is_ready or len(self._data) == 0:
            return None
        n   = min(n, len(self._data))
        idx = torch.randint(0, len(self._data), (n,))
        return self._data[idx].to(device=device, dtype=dtype, non_blocking=True)

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
    Periodic mining pass that populates a HardSampleBuffer with background
    windows the model currently ranks highest (false-alarm candidates).

    The mining pass runs in eval / inference mode; no gradient computation
    or weight update occurs.

    Thread-safety note
    ------------------
    MemmapNoiseSampler runs a prefetch thread.  We consume pre-fetched
    batches through sample_batch() — which only touches the thread-safe
    Queue — rather than calling _read_batch directly, so the prefetch
    thread can remain running throughout the mining pass.

    Memory note
    -----------
    Instead of accumulating all processed tensors in RAM and selecting
    top-K at the end, we prune to top-K every `prune_every` batches.

    Parameters
    ----------
    hard_noise_buffer : HardSampleBuffer
    n_mine_noise : int
        Approximate number of noise windows to evaluate per mining pass.
    prune_every : int
        Prune accumulated candidates to buffer capacity every this many batches.
    """

    def __init__(
        self,
        hard_noise_buffer: HardSampleBuffer,
        n_mine_noise:  int = 100_000,
        prune_every:   int = 20,
    ):
        self.hard_noise_buffer = hard_noise_buffer
        self.n_mine_noise      = n_mine_noise
        self.prune_every       = prune_every

    # ------------------------------------------------------------------
    @staticmethod
    def _streaming_top_k(
        new_data:    torch.Tensor,
        new_logits:  torch.Tensor,
        acc_data:    list,
        acc_logits:  list,
        capacity:    int,
    ):
        """Append new batch to accumulator lists, then prune to top-`capacity`."""
        acc_data.append(new_data.cpu().float())
        acc_logits.append(new_logits.cpu().float())

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

    # ------------------------------------------------------------------
    @torch.no_grad()
    def mine(
        self,
        model,
        noise_sampler,
        processor,
        device:   str,
        autocast: bool = False,
    ):
        """
        Run a mining pass over background windows and update the buffer.

        Uses noise_sampler.sample_batch() (thread-safe Queue consumer) so the
        prefetch thread can remain running throughout.
        """
        cfg  = get_cfg()
        bs   = cfg.batch_size
        cap  = self.hard_noise_buffer.capacity

        was_training = model.training
        model.eval()

        cast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if autocast else nullcontext()
        )

        print("  [Miner] Mining hard background windows …")
        acc_data, acc_logits = [], []
        n_batches = max(1, -(-self.n_mine_noise // bs))  # ceiling division

        for i in tqdm(range(n_batches), desc="  hard-bg", leave=False):
            noise_fd = noise_sampler.sample_batch()
            x        = processor(noise_fd)
            with cast:
                out = model(x)
            logits = out[0].reshape(-1).float().cpu()

            if (i + 1) % self.prune_every == 0 or i == n_batches - 1:
                self._streaming_top_k(x, logits, acc_data, acc_logits, cap)
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

        if was_training:
            model.train()
