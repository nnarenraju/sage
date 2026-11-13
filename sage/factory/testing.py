#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sage testing loop: separate noise and signal passes at any window count.

The noise pass gives the background tail that sets FAR thresholds and
records of the loudest noise windows, which :func:`query_testing`
selects and :func:`reconstruct_noise` re-reads from the memmaps.
"""

import time
import torch
import numpy as np

from sage.architecture.custom_losses import width_to_sigma
from sage.core.logger import get_logger, format_duration as _fmt_duration
from sage.factory.contract import forward_batch
from sage.utils.atomic_io import write_h5

logger = get_logger(__name__)


# Module-local name of the writer the search also uses
_write_h5 = write_h5


def far_threshold(top_noise, n_noise, far):
    """
    Ranking threshold whose noise exceedance rate is ``far`` per window.

    The threshold is the ``ceil(far * n_noise)``-th largest score; the
    largest when ``far < 1 / n_noise``, NaN when beyond the kept tail.

    Parameters
    ----------
    top_noise : array-like
        Descending high tail of the background scores.
    n_noise : int
        Number of background windows.
    far : float
        False-alarm rate per window.

    Returns
    -------
    float
        The threshold, or NaN.
    """
    rank = max(1, int(np.ceil(far * n_noise)))
    if rank > np.asarray(top_noise).size:
        return float("nan")
    return float(np.asarray(top_noise)[rank - 1])


def efficiency_at_far(top_noise, n_noise, sig_scores, sig_snr, far, snr_edges):
    """
    Detection efficiency vs SNR at a fixed FAR.

    Parameters
    ----------
    top_noise : array-like
        Descending high tail of the background scores.
    n_noise : int
        Number of background windows.
    sig_scores, sig_snr : array-like
        Ranking statistic and injected SNR of each injection.
    far : float
        False-alarm rate per window.
    snr_edges : numpy.ndarray
        SNR bin edges.

    Returns
    -------
    tuple of numpy.ndarray
        ``(snr_centres, efficiency, counts)``; ``efficiency[i]`` is the
        fraction of injections in bin ``i`` above ``far_threshold``,
        NaN for an empty bin.
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
    """
    Indices of the loud-noise records matching every given predicate.

    Parameters
    ----------
    h5path : str or Path
        File written by :meth:`SageVanillaTesting.run_noise`.
    far : float or None
        Keep records above the threshold of this FAR per window.
    score_above : float or None
        Keep records with ranking above this raw value.
    detector : str or None
        Keep records whose loudest detector (largest peak) is this one.
    peak_above : float or None
        Keep records whose largest per-detector peak exceeds this.
    pe_below, pe_above : dict of str to float, or None
        Keep records whose physical PE mean of each named head is
        below / above the value (heads as ``signal.targets``).

    Returns
    -------
    numpy.ndarray of int
        Indices into the ``loud_*`` record arrays.
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
    """
    Re-read the exact strain of loud-noise records from the memmaps.

    Parameters
    ----------
    h5path : str or Path
        File written by :meth:`SageVanillaTesting.run_noise`.
    sampler : MemmapNoiseSampler
        Live sampler built from the same run settings, exposing
        ``mmaps[d][run]`` and ``seq_len``.
    indices : array-like of int or None
        Records to read; None reads all.

    Returns
    -------
    numpy.ndarray
        ``(N, D, seq_len)`` float32 physical strain (stored values
        divided by the dynamic-range factor).
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


class SageVanillaTesting(torch.nn.Module):
    """
    Large-scale sensitivity testing: separate noise and signal passes.

    Parameters
    ----------
    signal_sampler, noise_sampler : torch.nn.Module
        Samplers; loud-noise records need the noise sampler's
        ``last_provenance``.
    processor : Preprocessor
        Preprocessing chain.
    model : torch.nn.Module
        The network.
    targets : list of str
        Estimated parameters, in head order (``signal.targets``).
    detectors : list of str
        Detectors, in channel order.
    width_bounds : tuple of float
        ``(sigma_min, sigma_max)`` of the loss's width mapping.
    device : str or torch.device
        Device of the batches.
    autocast_dtype : torch.dtype or None
        Autocast dtype on ``device``; None runs without autocast.
    keep_top, keep_loud, hist_bins : int
        Retained score tail, loudest noise windows kept as records, and
        histogram bins.
    """

    def __init__(
        self,
        signal_sampler,
        noise_sampler,
        processor,
        model,
        targets: list,
        detectors: list,
        width_bounds: tuple,
        device,
        autocast_dtype,
        keep_top=2_000_000,
        keep_loud=5_000,
        hist_bins=4000,
    ):
        super().__init__()

        self.detectors = list(detectors)
        self.width_bounds = tuple(width_bounds)
        self.device = torch.device(device)
        self.autocast_dtype = autocast_dtype

        self.signal_sampler = signal_sampler
        self.noise_sampler  = noise_sampler
        self.processor      = processor
        self.model          = model

        self.keep_top  = int(keep_top)      # retained score tail (FAR thresholds)
        self.keep_loud = int(keep_loud)     # loudest noise windows kept as records
        self.hist_bins = int(hist_bins)

        self.param_names = list(getattr(
            getattr(signal_sampler, "param_sampler", None), "param_names", []) or [])
        self.num_pe   = len(targets)
        self.pe_names = list(targets)

    def _forward(self, x):
        """
        Network output of a frequency-domain strain batch.

        Runs :func:`sage.factory.contract.forward_batch`.

        Parameters
        ----------
        x : torch.Tensor
            Complex ``(B, D, F)`` strain batch.

        Returns
        -------
        torch.Tensor
            ``(B, C)`` float32: column 0 the ranking statistic, then the
            raw PE means and sigmas.
        """
        return forward_batch(
            x,
            self.model,
            self.processor,
            amp_dtype=self.autocast_dtype,
            autocast=self.autocast_dtype is not None,
            device_type=self.device.type,
        )

    def _physical_pe(self, net):
        """
        Physical PE means and sigmas of a network output, as validation.

        Parameters
        ----------
        net : torch.Tensor
            Network output ``(B, C)``.

        Returns
        -------
        tuple of torch.Tensor
            ``(mu, sigma)``, each ``(B, P)`` float32; sigma is NaN for a
            mean-only model.
        """
        P = self.num_pe
        ps = self.signal_sampler.param_sampler
        mu_phys = ps.unscale_targets(net[:, 1:1 + P])
        if net.shape[1] >= 1 + 2 * P:
            sigma_std  = width_to_sigma(net[:, 1 + P:1 + 2 * P], *self.width_bounds)
            sigma_phys = sigma_std * ps._target_scales.to(sigma_std.device)
        else:
            sigma_phys = torch.full_like(mu_phys, float("nan"))
        return mu_phys.float(), sigma_phys.float()

    @torch.inference_mode()
    def run_noise(self, n_windows, save_path=None, log_every=500):
        """
        Score ``n_windows`` noise windows; keep records of the loudest.

        Keeps the descending score tail and a histogram, and, when the
        noise sampler gives ``last_provenance``, records of the
        ``keep_loud`` loudest windows (ranking, PE, peak, provenance).

        Parameters
        ----------
        n_windows : int
            Noise windows to score.
        save_path : str or Path or None
            HDF5 file for the result; None writes nothing.
        log_every : int
            Batches between progress lines; 0 for none.

        Returns
        -------
        dict
            ``top_noise``, ``n_noise``, ``noise_hist``, ``noise_edges``
            and, with provenance, the ``loud_*`` record arrays.
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
            # (B, D) per-detector loudness
            peak_np = noise.abs().amax(dim=2).cpu().numpy()
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
                     "detectors": ",".join(self.detectors),
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
        """
        Score ``n_windows`` signal-in-noise windows.

        Records the ranking, the injected SNR (the augment's last draw),
        the physical PE and every injection parameter.

        Parameters
        ----------
        n_windows : int
            Signal windows to score.
        save_path : str or Path or None
            HDF5 file for the result; None writes nothing.
        log_every : int
            Batches between progress lines; 0 for none.

        Returns
        -------
        dict
            ``sig_scores``, ``sig_snr``, ``sig_theta``, ``sig_pe_mu``
            and ``sig_pe_sigma``.

        Raises
        ------
        TypeError
            If ``signal_sampler.augment.target_snr_sampler`` records no
            ``last`` draw.
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
            # Signals rescaled to their target SNR
            hf, _, theta = self.signal_sampler(return_theta=True)
            last = sampler.last.detach().cpu().numpy()
            noise, _ = self.noise_sampler()
            # Signal and noise batches may differ in size
            nb = min(noise.shape[0], hf.shape[0])
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
                      attrs={"detectors": ",".join(self.detectors),
                             "param_names": ",".join(self.param_names),
                             "pe_names": ",".join(self.pe_names)})
            logger.info("[testing] wrote %s", save_path)
        return result
