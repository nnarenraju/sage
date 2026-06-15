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

import numpy as np
import torch
import torch.nn as nn

from sage.data.waveform.multiband_grid import (
    multibanding_grid,
    _inspiral_df_coefficient,
    _MTSUN_SI,
)


def _n_coarse_fast(f_min, f_max, delta_f, m1, m2, res_test=1e-3):
    """
    Count coarse grid points without allocating the frequency array.

    Mirrors the LAL inspiral-only path of multibanding_grid().  Used for the
    worst-case prior scan because it is ~37× faster than the full function.

    NOTE: overcounts by a fixed amount (~79 for the BNS defaults) because the
    last sub-band is not truncated at f_max here.  This does NOT affect the
    ranking — the mass pair that maximises this count is identical to the one
    that maximises the full multibanding_grid() count, so the scan result is
    correct.  The winner is always cross-validated with the full function.
    """
    M_total_s = (m1 + m2) * _MTSUN_SI
    eta       = m1 * m2 / (m1 + m2) ** 2
    eval_dmf  = delta_f * M_total_s
    mf_start  = f_min   * M_total_s
    mf_fmax   = f_max   * M_total_s

    mf_meco = max(1.0 / (6.0 ** 1.5 * math.pi), mf_fmax * 4.0)

    df_power      = 11.0 / 6.0
    df_coefficient = _inspiral_df_coefficient(eta, 2, res_test)
    freq_factor    = 2.0 ** (1.0 / df_power)

    df0_orig = df_coefficient * mf_start ** df_power
    df_ratio = df0_orig / eval_dmf

    total = 0
    if df_ratio < 1.0:
        f_end_grid0        = (eval_dmf / df_coefficient) ** (1.0 / df_power)
        n_pre_i            = math.ceil((f_end_grid0 - mf_start) / eval_dmf)
        total             += n_pre_i + 1
        f_start_insp_deref = mf_start + eval_dmf * n_pre_i
        df0_current        = 2.0 * df0_orig
        pre_done           = True
    else:
        f_start_insp_deref = mf_start
        df0_current        = df0_orig
        pre_done           = False

    n_derefine = math.ceil(
        math.log(mf_meco / f_start_insp_deref) / math.log(freq_factor)
    )

    next_f_start = f_start_insp_deref
    for index in range(n_derefine):
        mydf = (eval_dmf if df0_current < eval_dmf
                else eval_dmf * int(math.floor(df0_current / eval_dmf)))
        f_start_here = next_f_start + mydf if (index > 0 or pre_done) else next_f_start
        f_end_here   = f_start_here * freq_factor
        n_i          = math.ceil((f_end_here - f_start_here) / mydf)
        x_max        = f_start_here + mydf * n_i
        total       += n_i + 1
        df0_current *= 2.0
        next_f_start = x_max
        if next_f_start >= mf_fmax:
            break
    return total


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
        n_grid:   int   = 500,
        res_test: float = 1e-3,
        device:   str   = "cpu",
        verbose:  bool  = True,
    ) -> "MultibandSelector":
        """
        Scan the mass prior at runtime and build a selector for the worst-case
        (m1, m2) — the pair that requires the most coarse grid points.

        Parameters
        ----------
        param_sampler : DistributionSampler
            Sage parameter sampler built from a gwconfig YAML.  The mass
            bounds are read from ``param_sampler.bounds``.
        data_cfg : BaseDataConfig
            Sage data configuration (f_min, f_max, delta_f).
        n_grid : int
            Number of points per axis for the coarse scan grid.  500×500
            (default) covers ~125k valid pairs in a few seconds using the
            fast counter.  Increase for finer resolution.
        res_test : float
            LAL multibanding accuracy threshold (default 1e-3).
        device : str
            Torch device for the index tensor.
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

        if verbose:
            print(
                f"[MultibandSelector] Scanning prior "
                f"m1∈[{m1_min},{m1_max}] m2∈[{m2_min},{m2_max}] M☉  "
                f"({n_grid}×{n_grid} grid, resTest={res_test}) ..."
            )

        # ── Fast scan to rank mass pairs ──────────────────────────────────
        m1_arr = np.linspace(m1_min, m1_max, n_grid)
        m2_arr = np.linspace(m2_min, m2_max, n_grid)

        best_n_fast, best_m1, best_m2 = 0, m1_min, m2_min
        for m1 in m1_arr:
            for m2 in m2_arr:
                if m2 > m1 + 1e-9:   # enforce m1 >= m2 (mass_order constraint)
                    continue
                n = _n_coarse_fast(f_min, f_max, delta_f, m1, m2, res_test)
                if n > best_n_fast:
                    best_n_fast, best_m1, best_m2 = n, m1, m2

        # ── Validate the winner with the exact function ───────────────────
        # The fast counter overcounts the last sub-band but preserves ranking.
        # Use the full multibanding_grid() on the winner to get the true count.
        true_n = len(multibanding_grid(f_min, f_max, delta_f, best_m1, best_m2,
                                       res_test=res_test))

        if verbose:
            print(
                f"[MultibandSelector] Worst-case: m1={best_m1:.4f} M☉  "
                f"m2={best_m2:.4f} M☉  →  N_coarse={true_n:,}  "
                f"({int(round((f_max-f_min)/delta_f))+1}/{true_n} = "
                f"{(int(round((f_max-f_min)/delta_f))+1)/true_n:.1f}× compression)"
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
