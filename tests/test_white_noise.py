"""Tests for sage.data.noise.white_noise: curve checks and the batch sampler."""

import pytest
import torch

from sage.core.geometry import window_from_data
from sage.data.noise.white_noise import WhiteGaussianNoiseSampler, check_detector_asds

LIGO = "aLIGODesignSensitivityP1200087"
VIRGO = "AdVDesignSensitivityP1200087"


def test_asd_choice():
    assert check_detector_asds({"L1": LIGO, "H1": LIGO}, ["H1", "L1"]) == [LIGO, LIGO]
    with pytest.raises(ValueError, match=r"missing \['L1'\], extra \['V1'\]"):
        check_detector_asds({"H1": LIGO, "V1": VIRGO}, ["H1", "L1"])


def test_asd_wrong_curve():
    with pytest.raises(ValueError, match="did you mean 'aLIGODesignSensitivityP1200087'"):
        check_detector_asds({"H1": "aLIGODesignSensitivityP120087"}, ["H1"])
    with pytest.raises(ValueError, match="H1: 'AdVDesignSensitivityP1200087' is a Virgo curve, H1 is LIGO"):
        check_detector_asds({"H1": VIRGO}, ["H1"])


def test_sampler_reproducible():
    window = window_from_data(
        {"sample_rate_in_hz": 256.0, "sample_length_in_s": 1.0, "padding_length_in_s": 0.5}
    )

    def sampler():
        return WhiteGaussianNoiseSampler(
            [LIGO, VIRGO], ["H1", "V1"], window, 3, 7, 20.0, 1.0, "cpu", torch.float32
        )

    first, second = sampler(), sampler()
    batch, targets = first()
    assert batch.shape == (3, 2, window.n_frequency_bins) and batch.dtype == torch.complex64
    assert torch.equal(batch, second()[0])
    assert not torch.equal(first()[0], batch)
    first.set_step(0)
    assert torch.equal(first()[0], batch)
    assert torch.equal(targets, torch.zeros(3, 1))
