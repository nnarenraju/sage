"""Tests for sage.dsp.inverse_spectrum_truncation against PyCBC."""

import numpy as np
import pytest
import torch
from pycbc.psd import inverse_spectrum_truncation
from pycbc.types import FrequencySeries

from sage.dsp.inverse_spectrum_truncation import (
    inverse_spectrum_truncation_batch,
    inverse_spectrum_truncation_single,
)


@pytest.mark.parametrize("cutoff", [None, 20.0])
def test_matches_pycbc(cutoff):
    delta_f = 0.5
    freqs = np.arange(513) * delta_f
    psd = 1e-46 * (1.0 + (30.0 / np.maximum(freqs, 1.0)) ** 4 + (freqs / 150.0) ** 2)
    reference = inverse_spectrum_truncation(
        FrequencySeries(psd, delta_f=delta_f), 64, low_frequency_cutoff=cutoff,
        trunc_method="hann",
    ).numpy()
    single = inverse_spectrum_truncation_single(
        torch.from_numpy(psd), 64, low_frequency_cutoff=cutoff, delta_f=delta_f
    ).numpy()
    batch = inverse_spectrum_truncation_batch(
        torch.from_numpy(psd)[None, None], 64, low_frequency_cutoff=cutoff, delta_f=delta_f
    )[0, 0].numpy()
    above = freqs >= (cutoff or 0.0)
    above &= np.isfinite(reference)
    for out in (single, batch):
        np.testing.assert_allclose(out[above], reference[above], rtol=1e-12)
