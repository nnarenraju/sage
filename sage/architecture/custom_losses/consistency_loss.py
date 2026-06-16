#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Per-detector consistency loss.

A standalone, masked heteroscedastic Gaussian negative-log-likelihood for the
multi-detector consistency heads. Completely separate from
:class:`~sage.architecture.custom_losses.loss_functions.BCEWithPEsigmaLoss`
(the existing classification + merged-PE loss is untouched); the consistency
training loop adds this term on top.

The per-detector ``tc`` and ``mchirp`` heads are supervised with

    nll(mu, sigma, y) = 0.5 * (mu - y)^2 / sigma^2 + log(sigma)

masked per detector and averaged over the supervised entries. Detectors that
are not supervised in a given sample (mask = 0) contribute nothing, so their
sigma is free to grow — exactly the desired behaviour for noise detectors and
for the faint member of a faint coherent coincidence.

Masking convention (per-detector mask supplied by the caller, shape ``(B, D)``):
  - matched coherent          : 1 for both detectors
  - mismatch signal+noise     : 1 for the signal detector only
  - mismatch different-signal  : 1 for both (each toward its own truth)
  - pure noise                : 0 for both
The same mask gates both the ``tc`` and ``mchirp`` terms.
"""

import torch
import torch.nn as nn


class ConsistencyNLLLoss(nn.Module):
    """Masked per-detector Gaussian NLL for ``tc`` and ``mchirp``.

    Parameters
    ----------
    tc_weight, mc_weight : float
        Relative weights of the ``tc`` and ``mchirp`` NLL terms.
    eps : float
        Stabiliser for the mask-count denominator.

    Forward (all per-detector tensors are shape ``(B, D)``; targets broadcast
    from ``(B,)`` / ``(B, 1)``)
    -------
    mu_tc, sigma_tc, mu_mc, sigma_mc : predicted means / standard deviations
    tc_target : per-detector arrival times (window-normalised)
    mc_target : per-detector chirp mass (standardised)
    mask      : ``(B, D)`` per-detector supervision mask

    Returns
    -------
    torch.Tensor, shape ``(3,)`` : ``[total, tc_term, mchirp_term]`` (index 0 is
    the value to backpropagate, matching the convention of the other losses).
    """

    def __init__(
        self,
        tc_weight: float = 1.0,
        mc_weight: float = 1.0,
        beta: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.tc_weight = float(tc_weight)
        self.mc_weight = float(mc_weight)
        # beta-NLL exponent (Seitzer et al. 2022). 0 -> plain NLL; 0.5 recommended.
        self.beta = float(beta)
        self.eps = float(eps)
        self.num_components = 3  # [total, tc, mchirp]

    def _nll(self, mu, sigma, y):
        # float32 throughout: 1/sigma^2 can exceed fp16 range under autocast.
        # sigma is the actual std (softplus-parameterised, >= sigma_min), so
        # log(sigma) and 1/sigma^2 are finite without any exp.
        mu, sigma, y = mu.float(), sigma.float(), y.float()
        var = sigma * sigma
        nll = 0.5 * (mu - y) ** 2 / var + torch.log(sigma)
        if self.beta > 0.0:
            # beta-NLL: scale each term by stop_grad(sigma^(2 beta)). The plain
            # heteroscedastic gradient ~ 1/sigma^2 destabilises training when a
            # head turns briefly overconfident; this tempers it to ~1/sigma^(2-2b)
            # (beta=0.5 -> ~1/sigma) without biasing the optimum.
            nll = nll * (var ** self.beta).detach()
        return nll

    def forward(
        self,
        mu_tc,
        sigma_tc,
        mu_mc,
        sigma_mc,
        tc_target,
        mc_target,
        mask,
    ):
        if tc_target.dim() == 1:
            tc_target = tc_target.unsqueeze(1)
        if mc_target.dim() == 1:
            mc_target = mc_target.unsqueeze(1)

        nll_tc = self._nll(mu_tc, sigma_tc, tc_target)
        nll_mc = self._nll(mu_mc, sigma_mc, mc_target)

        mask = mask.float()
        denom = mask.sum() + self.eps
        loss_tc = (mask * nll_tc).sum() / denom
        loss_mc = (mask * nll_mc).sum() / denom
        total = self.tc_weight * loss_tc + self.mc_weight * loss_mc

        return torch.stack([total, loss_tc, loss_mc])
