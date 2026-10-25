"""Waveform parameter layout: needed parameters and LAL's refusals."""

import math

import pytest
import torch

from sage.data.waveform import waveform_utils as wu


def test_missing_parameter():
    index = {p: i for i, p in enumerate(wu.ALIGNED_PARAMETERS)}
    with pytest.raises(ValueError, match=r"IMRPhenomXAS_NRTidalv3 takes the waveform parameters \['lambda1', 'lambda2'\]"):
        wu.layout_columns(index, wu.TIDAL_ALIGNED_PARAMETERS, "IMRPhenomXAS_NRTidalv3")
    assert wu.given_extras(index, wu.ALIGNED_PARAMETERS) == ()


def test_refusals():
    theta = torch.full((2, len(wu.WAVEFORM_PARAMETERS)), math.nan)
    wu.refuse_nonzero(theta, wu.TRANSVERSE_SPIN_PARAMETERS + wu.TIDAL_PARAMETERS)  # absent: not given
    theta[:, wu.WAVEFORM_PARAMETERS.index("lambda1")] = torch.tensor([0.0, 300.0])
    with pytest.raises(ValueError, match="Non-zero tidal parameters were given"):
        wu.refuse_nonzero(theta, wu.TIDAL_PARAMETERS)
    theta[:, wu.WAVEFORM_PARAMETERS.index("spin2y")] = 0.1
    with pytest.raises(ValueError, match="Non-zero transverse spins were given"):
        wu.refuse_nonzero(theta, wu.TRANSVERSE_SPIN_PARAMETERS)
