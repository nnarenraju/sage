"""Tests for the merged classification+PE losses in custom_losses/loss_functions.

Covers BCEWithPEregLoss (BCE + smooth-L1 PE) and BCEWithPEsigmaLoss
(BCE + heteroscedastic beta-NLL + coupling), focusing on the NaN-proofing that
was the point of the sigma loss: softplus sigma floored at sigma_min / capped at
sigma_max (no exp), so extreme head outputs can't blow up 1/sigma^2. The loss
only reads cfg.do_point_estimate at construction, so get_cfg is patched — no
config registration, no GPU. CI-safe.
"""

from unittest.mock import patch

import pytest
import torch

from sage.architecture.custom_losses.loss_functions import (
    BCEWithPEregLoss,
    BCEWithPEsigmaLoss,
)

NUM_PE = 2                       # do_point_estimate = [tc, mchirp]
_CFG_PATH = "sage.architecture.custom_losses.loss_functions.get_cfg"


class _Cfg:
    do_point_estimate = ["tc", "mchirp"]


def _make(cls, **kw):
    with patch(_CFG_PATH, return_value=_Cfg()):
        return cls(**kw)


def _batch(B=8, sigma_loss=True, labels=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    rank = torch.randn(B, generator=g)
    pe_cols = 2 * NUM_PE if sigma_loss else NUM_PE
    pe = torch.randn(B, pe_cols, generator=g)
    if labels is None:
        labels = (torch.arange(B) % 2).float()          # alternating noise/signal
    targets = torch.cat([torch.randn(B, NUM_PE, generator=g), labels[:, None]], dim=1)
    return (rank, pe), targets


# ── BCEWithPEsigmaLoss ─────────────────────────────────────────────────────
class TestSigmaLoss:


    def test_noise_only_reg_is_zero(self):
        loss = _make(BCEWithPEsigmaLoss)
        out = loss(*_batch(labels=torch.zeros(8)))      # all noise -> no signal
        assert out[2].item() == 0.0                     # reg_loss masked to 0

    def test_sigma_bounds(self):
        loss = _make(BCEWithPEsigmaLoss, sigma_min=1e-3, sigma_max=10.0)
        raw = torch.tensor([-1e6, -20.0, 0.0, 20.0, 1e6])
        s = loss._sigma(raw)
        assert torch.isfinite(s).all()
        assert (s >= loss.sigma_min - 1e-9).all()
        assert (s <= loss.sigma_max + 1e-9).all()
        assert torch.isclose(s[0], torch.tensor(loss.sigma_min), atol=1e-6)   # floored
        assert torch.isclose(s[-1], torch.tensor(loss.sigma_max), atol=1e-6)  # capped

    @pytest.mark.parametrize("raw_sigma", [-1e6, -50.0, 50.0, 1e6])
    def test_nan_proof_extreme_sigma(self, raw_sigma):
        # extreme raw sigma (would be exp-blowup without softplus floor/cap)
        loss = _make(BCEWithPEsigmaLoss)
        B = 6
        rank = torch.randn(B)
        pe = torch.empty(B, 2 * NUM_PE)
        pe[:, :NUM_PE] = 1e3                              # large means too
        pe[:, NUM_PE:] = raw_sigma
        targets = torch.cat([torch.randn(B, NUM_PE), torch.ones(B, 1)], dim=1)
        out = loss((rank, pe), targets)
        assert torch.isfinite(out).all()


# ── BCEWithPEregLoss ───────────────────────────────────────────────────────
class TestRegLoss:


    def test_noise_only_reg_is_zero(self):
        loss = _make(BCEWithPEregLoss)
        out = loss(*_batch(sigma_loss=False, labels=torch.zeros(8)))
        assert out[2].item() == 0.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
