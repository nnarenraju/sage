"""
MultibandSelector: apply the LAL worst-case multibanding grid to any FD tensor.

The multibanding grid (from multiband_grid.py) produces coarse frequency points
that are exact integer multiples of the uniform grid spacing DELTA_F.  This
means multibanding is pure index selection — no interpolation, no approximation.

Mathematical guarantee:
    multiband(signal + noise)[i]  ==  multiband(signal)[i] + multiband(noise)[i]

because both sides compute h_fd[idx_i] + n_fd[idx_i] with identical float ops.

Usage
-----
    # Build from prior at runtime (recommended — no hardcoded masses):
    selector = MultibandSelector.from_prior_scan(param_sampler, data_cfg)

    # Or specify masses directly:
    selector = MultibandSelector.from_prior(m1_worst, m2_worst, data_cfg)

    hf_coarse  = selector(hf_full)    # (B, D, F_full) -> (B, D, N_coarse)
    nf_coarse  = selector(nf_full)
    injected   = hf_coarse + nf_coarse
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from sage.data.waveform.multiband_grid import multibanding_grid


# ── Exact worst-case scan (no approximation) ──────────────────────────────
#
# The worst-case scan counts the coarse points of the ACTUAL grid used by the
# pipeline — ``len(multibanding_grid(...))`` — for every mass pair.  There is no
# fast/approximate counter: the ranking is by the exact grid size.  The scan is
# distributed row-by-row across processes (identical exact math, just parallel).
#
# Worker state is populated once per process by the initializer so the m2 axis
# and constants are not re-pickled for every task.
_SCAN_CTX: dict = {}


def _init_scan_worker(m2_axis, f_min, f_max, delta_f, res_test):
    _SCAN_CTX["m2_axis"] = np.asarray(m2_axis, dtype=np.float64)
    _SCAN_CTX["const"]   = (float(f_min), float(f_max), float(delta_f), float(res_test))


def _scan_row(m1):
    """
    Exact worst case over all valid m2 (<= m1) for a fixed m1 — one grid row.

    Returns ``(best_n_coarse, m1, best_m2, n_evaluated)``.  Uses the real
    ``multibanding_grid()`` for every pair, so ``best_n_coarse`` is exact.
    """
    m2_axis = _SCAN_CTX["m2_axis"]
    f_min, f_max, delta_f, res_test = _SCAN_CTX["const"]
    best_n, best_m2, n_eval = 0, 0.0, 0
    for m2 in m2_axis:                       # m2_axis is sorted ascending
        if m2 > m1 + 1e-9:
            break                            # all further m2 violate m1 >= m2
        n = len(multibanding_grid(f_min, f_max, delta_f, m1, m2, res_test=res_test))
        n_eval += 1
        if n > best_n:
            best_n, best_m2 = n, float(m2)
    return best_n, float(m1), best_m2, n_eval


class MultibandSelector(nn.Module):
    """
    Select the worst-case multibanding indices from a full FD tensor.

    Parameters
    ----------
    coarse_indices : torch.LongTensor, shape (N_coarse,)
        Integer indices into the full FD array [0 .. F_full-1].
    coarse_freqs : torch.Tensor, shape (N_coarse,)
        Corresponding frequencies in Hz (for diagnostics).
    """

    def __init__(
        self,
        coarse_indices: torch.LongTensor,
        coarse_freqs:   torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("coarse_indices", coarse_indices)
        self.register_buffer("coarse_freqs",   coarse_freqs)

    @classmethod
    def from_prior(
        cls,
        m1_worst:  float,
        m2_worst:  float,
        data_cfg,
        res_test:  float = 1e-3,
        device:    str   = "cpu",
    ) -> "MultibandSelector":
        """
        Build a MultibandSelector for the given worst-case masses.

        Parameters
        ----------
        m1_worst, m2_worst : float
            Component masses (solar masses) that produce the most coarse grid
            points — the "worst case" for the prior.
        data_cfg : BaseDataConfig
            Sage data configuration (provides padded_delta_f, sample_rate, etc.)
        res_test : float
            LAL multibanding accuracy threshold (default 1e-3).
        device : str
            Torch device for the index tensor.
        """
        delta_f = data_cfg.padded_delta_f
        f_min   = data_cfg.signal_low_frequency_cutoff
        f_max   = data_cfg.sample_rate / 2.0

        coarse_freqs_np = multibanding_grid(
            f_min, f_max, delta_f, m1_worst, m2_worst, res_test=res_test
        )

        # Convert to integer indices into the full FD array [0..F_full-1].
        # All coarse points are exact integer multiples of delta_f (see
        # multiband_grid.py — intdfRatio is always an integer), so the
        # rounding residual is at machine-precision level (~1e-11).
        indices_np = np.round(coarse_freqs_np / delta_f).astype(np.int64)

        coarse_indices = torch.tensor(indices_np, dtype=torch.long,  device=device)
        coarse_freqs   = torch.tensor(coarse_freqs_np, dtype=torch.float64, device=device)

        return cls(coarse_indices, coarse_freqs)

    @classmethod
    def from_prior_scan(
        cls,
        param_sampler,
        data_cfg,
        min_samples: int   = 10_000_000,
        n_grid:      int | None = None,
        res_test:    float = 1e-3,
        device:      str   = "cpu",
        n_workers:   int | None = None,
        verbose:     bool  = True,
    ) -> "MultibandSelector":
        """
        Scan the mass prior at runtime and build a selector for the worst-case
        (m1, m2) — the pair that requires the most coarse grid points.

        The worst-case multibanding config is NOT the lowest chirp mass: LAL's
        multibanding places coarse points from a per-binary chirp-time envelope,
        and the pair that needs the finest grid (largest ``N_coarse``) lies at a
        prior-dependent band that must be found by search.  This method searches
        the full ``(m1, m2)`` support — the only parameters the grid depends on —
        at a resolution fine enough to evaluate at least ``min_samples`` valid
        mass pairs, then uses the winner's grid for the whole prior (a grid that
        is exact for the hardest binary is exact for all easier ones).

        EXACT: every pair is counted with the real ``multibanding_grid()`` (the
        same grid the pipeline uses) — there is no fast/approximate counter.  The
        scan is distributed across processes; the math is identical to a serial
        exact scan, only faster.

        Parameters
        ----------
        param_sampler : DistributionSampler
            Sage parameter sampler built from a gwconfig YAML.  The mass
            bounds are read from ``param_sampler.bounds``.
        data_cfg : BaseDataConfig
            Sage data configuration (f_min, f_max, delta_f).
        min_samples : int
            Minimum number of valid (m1 >= m2) mass pairs to evaluate in the
            scan (default 10,000,000).  The per-axis resolution ``n_grid`` is
            derived from this so the guarantee holds for any prior shape.
        n_grid : int or None
            Explicit per-axis grid resolution.  When None (default), it is
            derived from ``min_samples``.  Pass an int only to override.
        res_test : float
            LAL multibanding accuracy threshold (default 1e-3).
        device : str
            Torch device for the index tensor.
        n_workers : int or None
            Number of worker processes for the exact scan.  None (default) uses
            all available CPUs; 1 runs serially in-process.  The result does not
            depend on this — only the wall-clock time.
        verbose : bool
            Print scan progress and result.

        Returns
        -------
        MultibandSelector
            Selector built for the worst-case mass pair found in the prior.
        """
        # ── Extract mass bounds from the prior ────────────────────────────
        bounds = param_sampler.bounds
        if "mass1" not in bounds or "mass2" not in bounds:
            raise ValueError(
                "param_sampler.bounds must contain 'mass1' and 'mass2'. "
                f"Available keys: {list(bounds.keys())}"
            )
        m1_min, m1_max = float(bounds["mass1"][0]), float(bounds["mass1"][1])
        m2_min, m2_max = float(bounds["mass2"][0]), float(bounds["mass2"][1])

        f_min   = float(data_cfg.signal_low_frequency_cutoff)
        f_max   = float(data_cfg.sample_rate / 2.0)
        delta_f = float(data_cfg.padded_delta_f)

        # ── Derive per-axis resolution from the sample-count target ────────
        # The m1 >= m2 constraint censors ~half of a square grid when the m1
        # and m2 ranges coincide (the conservative worst case), so target
        # 2*min_samples total grid points to guarantee >= min_samples valid.
        if n_grid is None:
            n_grid = int(math.ceil(math.sqrt(2.0 * float(min_samples))))

        if n_workers is None:
            n_workers = os.cpu_count() or 1

        if verbose:
            print(
                f"[MultibandSelector] EXACT scan of prior "
                f"m1∈[{m1_min},{m1_max}] m2∈[{m2_min},{m2_max}] M☉  "
                f"({n_grid}×{n_grid} grid, Δm≈{(m1_max-m1_min)/max(n_grid-1,1):.2e} M☉, "
                f"target≥{min_samples:,} valid pairs, resTest={res_test}, "
                f"{n_workers} worker(s)) ..."
            )

        # ── Exact scan: count the REAL multibanding grid for every pair ────
        # Distributed row-by-row (fixed m1, all valid m2) across processes.
        # Every count is len(multibanding_grid(...)) — no approximation.
        m1_axis = np.linspace(m1_min, m1_max, n_grid)
        m2_axis = np.linspace(m2_min, m2_max, n_grid)

        n_valid = 0
        best_n, best_m1, best_m2 = 0, m1_min, m2_min

        if n_workers <= 1:
            _init_scan_worker(m2_axis, f_min, f_max, delta_f, res_test)
            for m1 in m1_axis:
                bn, bm1, bm2, ne = _scan_row(float(m1))
                n_valid += ne
                if bn > best_n:
                    best_n, best_m1, best_m2 = bn, bm1, bm2
        else:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_scan_worker,
                initargs=(m2_axis, f_min, f_max, delta_f, res_test),
            ) as ex:
                for bn, bm1, bm2, ne in ex.map(
                    _scan_row, [float(x) for x in m1_axis], chunksize=16
                ):
                    n_valid += ne
                    if bn > best_n:
                        best_n, best_m1, best_m2 = bn, bm1, bm2

        if verbose:
            n_uniform = int(round((f_max - f_min) / delta_f)) + 1
            print(
                f"[MultibandSelector] Scanned {n_valid:,} valid mass pairs (EXACT).  "
                f"Worst-case: m1={best_m1:.5f} M☉  "
                f"m2={best_m2:.5f} M☉  →  N_coarse={best_n:,}  "
                f"({n_uniform}/{best_n} = {n_uniform/best_n:.1f}× compression)"
            )

        return cls.from_prior(
            m1_worst=best_m1,
            m2_worst=best_m2,
            data_cfg=data_cfg,
            res_test=res_test,
            device=device,
        )

    @property
    def n_coarse(self) -> int:
        return int(self.coarse_indices.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Select coarse frequency bins from a full FD tensor.

        Parameters
        ----------
        x : torch.Tensor
            Shape (..., F_full).  The last dimension is the full FD axis.

        Returns
        -------
        torch.Tensor
            Shape (..., N_coarse).  Identical float values as x at the
            selected indices — no arithmetic, just index gather.
        """
        return x[..., self.coarse_indices]
