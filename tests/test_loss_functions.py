"""Tests for the classification + parameter-estimation losses.

The point of ``BCE_PE_Sigma`` is NaN-proofing: softplus widths floored at
``sigma_min`` and capped at ``sigma_max`` (no exp), so extreme head outputs
cannot blow up ``1/sigma^2``. Components are named per target.
"""

import pytest
import torch

from sage.architecture.custom_losses import BCE_PE_Huber, BCE_PE_MSE, BCE_PE_Sigma, width_to_sigma

TARGETS = ["delta_tc", "mchirp"]


def batch(widths=True, labels=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    rank = torch.randn(8, 1, generator=g)
    estimates = torch.randn(8, 4 if widths else 2, generator=g)
    if labels is None:
        labels = (torch.arange(8) % 2).float()          # alternating noise/signal
    targets = torch.cat([torch.randn(8, 2, generator=g), labels[:, None]], dim=1)
    return (rank, estimates, torch.zeros(8, 0)), targets


@pytest.mark.parametrize(
    "loss, widths",
    [(BCE_PE_Sigma(TARGETS), True), (BCE_PE_Huber(TARGETS), False), (BCE_PE_MSE(TARGETS), False)],
)
def test_components(loss, widths):
    out = loss(*batch(widths))
    names = loss.component_names
    assert len(out) == loss.num_components == len(names)
    assert names[:4] == ["total", "classification", "regression_delta_tc", "regression_mchirp"]
    regression = out[2] + out[3]
    coupling = loss.coupling_weight * out[4] if "coupling" in names else 0.0
    assert torch.isclose(out[0], out[1] + loss.regression_weight * regression + coupling)


def test_noise_only_regression_is_zero():
    out = BCE_PE_Sigma(TARGETS)(*batch(labels=torch.zeros(8)))
    assert (out[2:4] == 0.0).all()


def test_confidence_weighting():
    outputs, targets = batch(widths=False)
    plain = BCE_PE_MSE(TARGETS, confidence_weighting=False)(outputs, targets)
    weighted = BCE_PE_MSE(TARGETS, confidence_weighting=True)(outputs, targets)
    assert torch.equal(plain[1], weighted[1])          # classification unchanged
    assert (weighted[2:] < plain[2:]).all()            # each term scaled by p < 1


def test_non_astrophysical_class():
    """Multinomial cross-entropy over the logits (0, r, a); class-2 samples get no regression, like noise."""
    (rank, estimates, _), targets = batch(labels=torch.tensor([0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0]))
    logit = torch.randn(8, 1, generator=torch.Generator().manual_seed(1))
    out = BCE_PE_Sigma(TARGETS)((rank, estimates, logit), targets)
    logits = torch.cat([torch.zeros(8, 1), rank, logit], dim=1)
    assert torch.isclose(out[1], torch.nn.functional.cross_entropy(logits, targets[:, -1].long()))
    as_noise = targets.clone()
    as_noise[:, -1] = (targets[:, -1] == 1).float()
    assert torch.equal(BCE_PE_Sigma(TARGETS)((rank, estimates, logit), as_noise)[2:], out[2:])


def test_sigma_bounds():
    s = width_to_sigma(torch.tensor([-1e6, -20.0, 0.0, 20.0, 1e6]), 1e-3, 10.0)
    assert torch.isfinite(s).all()
    assert torch.isclose(s[0], torch.tensor(1e-3), atol=1e-6)   # floored
    assert torch.isclose(s[-1], torch.tensor(10.0), atol=1e-6)  # capped


@pytest.mark.parametrize("raw_width", [-1e6, -50.0, 50.0, 1e6])
def test_nan_proof_extreme_width(raw_width):
    rank = torch.randn(6, 1)
    estimates = torch.empty(6, 4)
    estimates[:, :2] = 1e3                              # large means too
    estimates[:, 2:] = raw_width
    targets = torch.cat([torch.randn(6, 2), torch.ones(6, 1)], dim=1)
    assert torch.isfinite(BCE_PE_Sigma(TARGETS)((rank, estimates, torch.zeros(6, 0)), targets)).all()
