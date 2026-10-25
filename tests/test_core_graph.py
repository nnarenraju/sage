"""Tests for sage.core.graph: the preprocessing chain and its order check."""

import numpy as np
import pytest
import torch

from sage.core.geometry import window_from_data
from sage.core.graph import Preprocessor
from sage.core.pipeline import GWBatch, Grid, PipelineError, ProcessingState
from sage.dsp.multirate_sampling import MultirateSampler
from sage.dsp.normalise import MinMaxNormalise
from sage.dsp.whiten import FiducialWhitening

WINDOW = window_from_data(
    {"sample_rate_in_hz": 256.0, "sample_length_in_s": 2.0, "padding_length_in_s": 0.5}
)
FD = ProcessingState(Grid.FD_UNIFORM)


def steps():
    whiten = FiducialWhitening(torch.ones(1, WINDOW.n_frequency_bins), WINDOW, "cpu")
    bins = np.array([[0, WINDOW.sample_length_in_samples, 256]])
    return whiten, MinMaxNormalise(), MultirateSampler(bins, WINDOW, "cpu", torch.float32)


def fd_batch(state):
    return GWBatch(torch.zeros(2, 1, WINDOW.n_frequency_bins, dtype=torch.complex64), state)


def test_chain_output_state():
    chain = Preprocessor(list(steps()), FD)
    assert chain.output_state == ProcessingState(Grid.TD_MULTIRATE, whitened=True)
    batch = chain(fd_batch(FD))
    assert batch.state == chain.output_state
    assert batch.data.shape == (2, 1, WINDOW.sample_length_in_samples)


@pytest.mark.parametrize(
    "order, state, message",
    [
        ((2, 0), FD, r"preprocess\[0\] \(MultirateSampler\)"),
        ((0, 0), FD, r"preprocess\[1\] \(FiducialWhitening\): .*already whitened"),
        ((1, 0), FD, r"preprocess\[0\] \(MinMaxNormalise\): .*time-domain"),
        ((0, 2), ProcessingState(Grid.TD_UNIFORM), r"preprocess\[0\] \(FiducialWhitening\): .*only FD_UNIFORM"),
    ],
)
def test_chain_order(order, state, message):
    available = steps()
    with pytest.raises(PipelineError, match=message):
        Preprocessor([available[i] for i in order], state)


def test_batch_state():
    whiten, _, multirate = steps()
    chain = Preprocessor([whiten, multirate], FD)
    with pytest.raises(PipelineError, match="built for"):
        chain(fd_batch(ProcessingState(Grid.FD_UNIFORM, whitened=True)))
