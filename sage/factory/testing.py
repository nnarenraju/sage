#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sage testing loop.

SageVanillaTesting
    Large-scale sensitivity testing, shaped like SageVanillaValidation but split
    into two INDEPENDENT passes, each run on demand at whatever window count is
    asked for (rather than one training-sized, fixed class-balance loop):

      * ``run_noise(n)``  -- pure noise windows -> ranking scores. The empirical
                             background; its high quantiles set the false-alarm-
                             rate (FAR) threshold. No waveform generation, so it
                             scales to O(1e8) windows for a low FAR (1 / 6 months).
      * ``run_signal(n)`` -- one injection per window (signal added to noise) ->
                             (ranking score, injected SNR, injection params).

    For the ``keep_loud`` loudest noise windows -- the ones that set the lowest
    FARs -- a compact per-window RECORD is stored (ranking, physical PE mean/sigma,
    per-detector peak, and per-detector run/segment/start provenance) rather than
    raw strain. :func:`reconstruct_noise` re-reads the exact strain from the noise
    memmaps on demand; :func:`query_testing` selects records by FAR / detector /
    PE / peak. Efficiency vs SNR at a fixed FAR is :func:`efficiency_at_far`.

    Confound-free comparisons are the caller's to arrange -- this class does not
    couple the SNR reference to the whitening. Rescale signals to a chosen SNR
    reference via the signal sampler's ``augment`` (an ``OptimalSNRRescaler``
    whose estimator ASD you set), and whiten with the model's own fiducial via
    ``processor``. The injected SNR per window is read from
    ``signal_sampler.augment.target_snr_sampler.last``.
"""

import os
import time
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

# LOCAL
from sage.core.config import get_cfg
from sage.core.logger import get_logger, format_duration as _fmt_duration
from sage.core.pipeline import GWBatch, Grid, ProcessingState
from sage.factory.contract import forward_batch
from sage.utils.atomic_io import write_h5

logger = get_logger(__name__)


# ======================================================================= io/query
# Kept as a module-local name for the existing call sites; the implementation is shared
# with the search, which writes its trigger shards and manifests the same way.
_write_h5 = write_h5


def far_threshold(top_noise, n_noise, far):
    """Ranking threshold whose noise exceedance rate is ``far`` (per window).

    ``top_noise`` is the descending high tail of ``n_noise`` background scores; the
    threshold is the ``ceil(far * n_noise)``-th largest. ``far`` below ``1/n_noise``
    is not resolvable (returns the largest score); a ``far`` needing a deeper tail
    than retained returns NaN.
    """
    rank = max(1, int(np.ceil(far * n_noise)))
    if rank > np.asarray(top_noise).size:
        return float("nan")
    return float(np.asarray(top_noise)[rank - 1])


def efficiency_at_far(top_noise, n_noise, sig_scores, sig_snr, far, snr_edges):
    """Detection efficiency vs SNR at a fixed FAR.

    Returns ``(snr_centres, efficiency, counts)``; ``efficiency[i]`` is the
    fraction of injections in SNR bin ``i`` scoring above ``far_threshold``.
    """
    thr = far_threshold(top_noise, n_noise, far)
    centres = 0.5 * (snr_edges[:-1] + snr_edges[1:])
    eff = np.full(centres.size, np.nan)
    cnt = np.zeros(centres.size, dtype=np.int64)
    which = np.digitize(sig_snr, snr_edges) - 1
    for i in range(centres.size):
        m = which == i
        cnt[i] = int(m.sum())
        if cnt[i]:
            eff[i] = float((np.asarray(sig_scores)[m] > thr).mean())
    return centres, eff, cnt


def query_testing(h5path, far=None, score_above=None, detector=None,
                  peak_above=None, pe_below=None, pe_above=None):
    """Select loud-noise record indices matching predicates.

    Parameters
    ----------
    far : float
        Keep records louder than the FAR threshold (i.e. false alarms at <= far).
    score_above : float
        Keep records with ranking above this raw value.
    detector : str
        Keep records whose LOUDEST detector (max per-detector peak) is this one.
    peak_above : float
        Keep records whose max per-detector peak exceeds this.
    pe_below, pe_above : dict[str, float]
        Keep records whose physical PE mean for the named head is below / above
        the value (head names are ``cfg.do_point_estimate``, e.g. "tc","mchirp").

    Returns
    -------
    np.ndarray[int]  indices into the ``loud_*`` record arrays.
    """
    import h5py
    with h5py.File(h5path, "r") as f:
        s   = f["loud_scores"][:]
        mu  = f["loud_pe_mu"][:] if "loud_pe_mu" in f else None
        pk  = f["loud_peak"][:] if "loud_peak" in f else None
        top = f["top_noise"][:]
        n_noise  = int(f.attrs["n_noise"])
        pe_names = list(str(f.attrs.get("pe_names", "")).split(",")) if f.attrs.get("pe_names") else []
        dets     = list(str(f.attrs.get("detectors", "")).split(","))
    mask = np.ones(s.shape[0], dtype=bool)
    if far is not None:
        mask &= s > far_threshold(top, n_noise, far)
    if score_above is not None:
        mask &= s > score_above
    if peak_above is not None and pk is not None:
        mask &= pk.max(axis=1) > peak_above
    if detector is not None and pk is not None:
        mask &= pk.argmax(axis=1) == dets.index(detector)
    for d, op in ((pe_below, np.less), (pe_above, np.greater)):
        if d and mu is not None:
            for name, val in d.items():
                mask &= op(mu[:, pe_names.index(name)], val)
    return np.nonzero(mask)[0]


def reconstruct_noise(h5path, sampler, indices=None):
    """Re-read the exact strain of loud-noise records from the noise memmaps.

    ``sampler`` is a live ``MemmapNoiseSampler`` (built from the same run config)
    exposing ``mmaps[d][run]`` and ``seq_len``. Returns ``(N, D, seq_len)`` float32
    physical strain (dyn-range restored), matching what the model was scored on.
    """
    import h5py
    from sage.data.noise._pycbc_lazy import dyn_range_fac
    with h5py.File(h5path, "r") as f:
        run   = f["loud_run"][:]
        start = f["loud_start"][:]
    if indices is None:
        indices = np.arange(run.shape[0])
    indices = np.asarray(indices)
    D, L = run.shape[1], int(sampler.seq_len)
    out = np.empty((indices.size, D, L), dtype=np.float32)
    scale = dyn_range_fac()
    for i, k in enumerate(indices):
        for d in range(D):
            r, s0 = int(run[k, d]), int(start[k, d])
            out[i, d] = np.asarray(sampler.mmaps[d][r][s0:s0 + L]) / scale
    return out


# ==================================================================== testing loop
class SageVanillaTesting(torch.nn.Module):
    """Large-scale sensitivity testing loop (see module docstring)."""

    def __init__(
        self,
        signal_sampler,
        noise_sampler,
        processor,
        model,
        amp_dtype=torch.float16,
        keep_top=2_000_000,
        keep_loud=5_000,
        hist_bins=4000,
    ):
        super().__init__()

        self.cfg = get_cfg()
        self.amp_dtype = amp_dtype

        self.signal_sampler = signal_sampler
        self.noise_sampler  = noise_sampler
        self.processor      = processor
        self.model          = model

        self.keep_top  = int(keep_top)      # retained score tail (FAR thresholds)
        self.keep_loud = int(keep_loud)     # loudest noise windows kept as records
        self.hist_bins = int(hist_bins)

        self.param_names = list(getattr(
            getattr(signal_sampler, "param_sampler", None), "param_names", []) or [])
        self.num_pe   = len(self.cfg.do_point_estimate)
        self.pe_names = list(self.cfg.do_point_estimate)

        # Multiband parity with SageVanillaValidation (selector optional).
        self._initial_state = getattr(
            signal_sampler, "output_state", ProcessingState(Grid.FD_UNIFORM))
        self._selector       = getattr(signal_sampler, "selector", None)
        self._freqs          = None
        self._coarse_indices = None
        if self._selector is not None:
            self._freqs          = self._selector.coarse_freqs
            self._coarse_indices = self._selector.coarse_indices

    # ------------------------------------------------------------------ core
    def _forward(self, x):
        """Raw strain batch ``(B, D, F)`` -> full network output ``(B, C)``.

        Mirrors SageVanillaValidation's forward path (selector -> GWBatch ->
        processor -> autocast model). Column 0 is the ranking statistic; the rest
        are the raw PE means then sigmas.

        The path itself lives in ``sage.factory.contract`` so that testing,
        benchmarking and the search all run the identical contract rather than
        three copies that can drift.
        """
        return forward_batch(
            x,
            self.model,
            self.processor,
            state=self._initial_state,
            selector=self._selector,
            freqs=self._freqs,
            coarse_indices=self._coarse_indices,
            amp_dtype=self.amp_dtype,
            autocast=self.cfg.autocast,
        )

    def _physical_pe(self, net):
        """Full output ``(B, C)`` -> physical PE mean/sigma ``(B, P)`` (as validation)."""
        P = self.num_pe
        ps = self.signal_sampler.param_sampler
        mu_std = net[:, 1:1 + P]
        if getattr(self.cfg, "pe_target_minmax", False):
            mu_phys = ps.unnorm_from_batch(mu_std)
        else:
            mu_phys = ps.unstandardise_from_batch(mu_std)
        if net.shape[1] >= 1 + 2 * P:
            sigma_std  = F.softplus(net[:, 1 + P:1 + 2 * P]) + 1e-3
            sigma_phys = sigma_std * ps._std_stds.to(sigma_std.device)
        else:
            sigma_phys = torch.full_like(mu_phys, float("nan"))
        return mu_phys.float(), sigma_phys.float()

    # ----------------------------------------------------------------- passes
    @torch.inference_mode()
    def run_noise(self, n_windows, save_path=None, log_every=500):
        """Score ``n_windows`` pure-noise windows; keep loud-window records.

        Stores the descending score tail + histogram (FAR thresholds), and for the
        ``keep_loud`` loudest windows a record of {ranking, physical PE mean/sigma,
        per-detector peak, per-detector run/segment/start}. Needs a noise sampler
        exposing ``last_provenance`` (TestNoiseSampler).
        """
        self.model.eval()
        n_windows = int(n_windows)
        scores = np.empty(n_windows, dtype=np.float32)

        keep = int(min(self.keep_loud, n_windows))
        P = self.num_pe
        L = {"s": np.full(keep, -np.inf, np.float32)}      # loud buffer (lazily grown)
        L["mu"] = L["sigma"] = L["peak"] = L["run"] = L["seg"] = L["start"] = None
        loud_min = -np.inf

        p, t0 = 0, time.time()
        while p < n_windows:
            noise, _ = self.noise_sampler()                    # (B, D, F) complex
            prov = getattr(self.noise_sampler, "last_provenance", None)
            net = self._forward(noise)                         # (B, C)
            mu, sigma = self._physical_pe(net)                 # (B, P) each
            s = net[:, 0].cpu().numpy()
            mu_np, sig_np = mu.cpu().numpy(), sigma.cpu().numpy()
            peak_np = noise.abs().amax(dim=2).cpu().numpy()    # (B, D) per-detector loudness
            k = min(s.shape[0], n_windows - p)
            scores[p:p + k] = s[:k]

            if prov is not None:
                Dd = prov["run"].shape[1]
                if L["run"] is None:
                    L["mu"]    = np.zeros((keep, P), np.float32)
                    L["sigma"] = np.zeros((keep, P), np.float32)
                    L["peak"]  = np.zeros((keep, Dd), np.float32)
                    L["run"]   = np.zeros((keep, Dd), np.int64)
                    L["seg"]   = np.zeros((keep, Dd), np.int64)
                    L["start"] = np.zeros((keep, Dd), np.int64)
                s_b = s[:k]
                cand = (np.arange(k) if not np.isfinite(loud_min)
                        else np.nonzero(s_b > loud_min)[0])
                if cand.size:
                    m_s   = np.concatenate([L["s"],   s_b[cand]])
                    m_mu  = np.concatenate([L["mu"],  mu_np[:k][cand]], axis=0)
                    m_sg  = np.concatenate([L["sigma"], sig_np[:k][cand]], axis=0)
                    m_pk  = np.concatenate([L["peak"], peak_np[:k][cand]], axis=0)
                    m_run = np.concatenate([L["run"], prov["run"][:k][cand]], axis=0)
                    m_seg = np.concatenate([L["seg"], prov["segment"][:k][cand]], axis=0)
                    m_st  = np.concatenate([L["start"], prov["start"][:k][cand]], axis=0)
                    top = np.argpartition(m_s, -keep)[-keep:]
                    L["s"], L["mu"], L["sigma"], L["peak"] = m_s[top], m_mu[top], m_sg[top], m_pk[top]
                    L["run"], L["seg"], L["start"] = m_run[top], m_seg[top], m_st[top]
                    loud_min = float(L["s"].min())

            p += k
            if log_every and (p // max(1, s.shape[0])) % log_every == 0:
                rate = p / max(1e-9, time.time() - t0)
                logger.info("[testing] noise %s/%s  %.0f/s  eta %s",
                            f"{p:,}", f"{n_windows:,}", rate,
                            _fmt_duration((n_windows - p) / max(1.0, rate)))
        logger.info("[testing] NOISE done %s in %s",
                    f"{n_windows:,}", _fmt_duration(time.time() - t0))

        topk = int(min(n_windows, self.keep_top))
        part = np.partition(scores, n_windows - topk)[n_windows - topk:]
        top_noise = np.sort(part)[::-1].astype(np.float32)
        noise_hist, noise_edges = np.histogram(scores, bins=self.hist_bins)

        result = dict(top_noise=top_noise, n_noise=np.int64(n_windows),
                      noise_hist=noise_hist.astype(np.int64),
                      noise_edges=noise_edges.astype(np.float32))
        if L["run"] is not None:
            order = np.argsort(L["s"])[::-1]
            result.update(loud_scores=L["s"][order].astype(np.float32),
                          loud_pe_mu=L["mu"][order], loud_pe_sigma=L["sigma"][order],
                          loud_peak=L["peak"][order], loud_run=L["run"][order],
                          loud_segment=L["seg"][order], loud_start=L["start"][order])
        if save_path:
            files = getattr(self.noise_sampler, "bin_files", None)
            attrs = {"n_noise": int(n_windows),
                     "detectors": ",".join(self.cfg.detectors),
                     "pe_names": ",".join(self.pe_names),
                     "seq_len": int(getattr(self.noise_sampler, "seq_len", 0))}
            if files:
                attrs["noise_files"] = ",".join(str(f) for f in files)
            _write_h5(save_path,
                      {kk: vv for kk, vv in result.items() if kk != "n_noise"}, attrs)
            logger.info("[testing] wrote %s", save_path)
        return result

    @torch.inference_mode()
    def run_signal(self, n_windows, save_path=None, log_every=500):
        """Score ``n_windows`` signal-in-noise windows.

        Records {ranking, injected SNR, physical PE mean/sigma, full injection
        params ``all_theta``} so misses slice by loudness, PE error, or intrinsic
        parameters (masses/spins/sky). SNR is the augment's recorded last draw.
        """
        self.model.eval()
        aug = getattr(self.signal_sampler, "augment", None)
        sampler = getattr(aug, "target_snr_sampler", None)
        if sampler is None or not hasattr(sampler, "last"):
            raise TypeError(
                "SageVanillaTesting.run_signal needs "
                "signal_sampler.augment.target_snr_sampler with a recorded `.last`.")
        n_windows = int(n_windows)
        P = self.num_pe
        scores = np.empty(n_windows, dtype=np.float32)
        snr    = np.empty(n_windows, dtype=np.float32)
        pe_mu  = np.empty((n_windows, P), dtype=np.float32)
        pe_sig = np.empty((n_windows, P), dtype=np.float32)
        theta_arr = None
        p, t0 = 0, time.time()
        while p < n_windows:
            hf, _, theta = self.signal_sampler(return_theta=True)   # rescaled to target SNR
            last = sampler.last.detach().cpu().numpy()
            noise, _ = self.noise_sampler()
            nb = min(noise.shape[0], hf.shape[0])          # align (signal batch = B*class_balance)
            net = self._forward(noise[:nb] + hf[:nb])
            mu, sigma = self._physical_pe(net)
            s = net[:, 0].cpu().numpy()
            th = theta[:nb].detach().cpu().numpy().astype(np.float32)
            if theta_arr is None:
                theta_arr = np.empty((n_windows, th.shape[1]), dtype=np.float32)
            k = min(nb, n_windows - p)
            scores[p:p + k] = s[:k]
            snr[p:p + k]    = last[:k]
            pe_mu[p:p + k]  = mu.cpu().numpy()[:k]
            pe_sig[p:p + k] = sigma.cpu().numpy()[:k]
            theta_arr[p:p + k] = th[:k]
            p += k
            if log_every and (p // max(1, nb)) % log_every == 0:
                rate = p / max(1e-9, time.time() - t0)
                logger.info("[testing] signal %s/%s  %.0f/s", f"{p:,}", f"{n_windows:,}", rate)
        logger.info("[testing] SIGNAL done %s in %s",
                    f"{n_windows:,}", _fmt_duration(time.time() - t0))

        result = dict(sig_scores=scores, sig_snr=snr, sig_theta=theta_arr,
                      sig_pe_mu=pe_mu, sig_pe_sigma=pe_sig)
        if save_path:
            _write_h5(save_path, result,
                      attrs={"detectors": ",".join(self.cfg.detectors),
                             "param_names": ",".join(self.param_names),
                             "pe_names": ",".join(self.pe_names)})
            logger.info("[testing] wrote %s", save_path)
        return result
