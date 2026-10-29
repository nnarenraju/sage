"""Every waveform model vs lalsimulation.SimInspiralChooseFDWaveform (multibanding off) on the production grids.

Where LAL is non-zero, each model holds its measured agreement rounded up to the next decade (2026-10-29); where
LAL is zero, every model is zero. IMRPhenomXHM and IMRPhenomXPHM carry their known, unresolved gap.
"""

from pathlib import Path

import lal
import lalsimulation as ls
import numpy as np
import pytest
import torch

from sage.config.prior import load_prior
from sage.data.waveform import waveform_utils
from sage.data.waveform.approximants.IMRPhenomD import IMRPhenomD
from sage.data.waveform.approximants.IMRPhenomXAS import IMRPhenomXAS
from sage.data.waveform.approximants.IMRPhenomXAS_NRTidalv3 import IMRPhenomXAS_NRTidalv3
from sage.data.waveform.approximants.IMRPhenomXHM import IMRPhenomXHM
from sage.data.waveform.approximants.IMRPhenomXPHM import IMRPhenomXPHM
from sage.factory.build import build_data, build_signal

F_LOW, RATE, T = 20.0, 2048.0, 16.0
COMMON = dict(distance=400.0, inclination=0.5, coa_phase=0.3)
ALIGNED = [dict(mass1=30.0, mass2=20.0, spin1z=0.3, spin2z=-0.2),
           dict(mass1=50.0, mass2=7.0, spin1z=0.9, spin2z=0.8),
           dict(mass1=10.0, mass2=10.0),
           dict(mass1=7.0, mass2=7.0, spin1z=-0.95, spin2z=-0.5)]
PRECESSING = [dict(mass1=30.0, mass2=20.0, spin1x=0.5, spin1y=0.3, spin1z=0.4, spin2x=-0.2, spin2y=0.1, spin2z=-0.3),
              dict(mass1=50.0, mass2=7.0, spin1x=0.7, spin1y=-0.4, spin1z=0.2, spin2x=0.1, spin2y=0.5, spin2z=0.6),
              dict(mass1=12.0, mass2=9.0, spin1x=0.05, spin1z=0.1),
              dict(mass1=40.0, mass2=35.0, spin1x=-0.6, spin1y=0.6, spin1z=-0.3, spin2x=0.4, spin2y=-0.4, spin2z=0.5)]
BNS = [dict(mass1=1.6, mass2=1.3, spin1z=0.03, spin2z=-0.02, lambda1=400.0, lambda2=900.0, distance=40.0),
       dict(mass1=1.4, mass2=1.4, lambda1=300.0, lambda2=300.0, distance=40.0, inclination=1.0, coa_phase=1.1)]


def theta_of(signals):
    """Signals in the WAVEFORM_PARAMETERS layout (absent spins 0, absent tides NaN)."""
    names = waveform_utils.WAVEFORM_PARAMETERS
    theta = torch.full((len(signals), len(names)), float("nan"), dtype=torch.float64)
    for row, s in enumerate(signals):
        values = {**COMMON, **s}
        for column, name in enumerate(names):
            theta[row, column] = values.get(name, 0.0 if name.startswith("spin") else float("nan"))
    return theta


def lal_hphc(s, approximant, delta_f):
    s = {**COMMON, **s}
    p = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(p, 0)
    ls.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(p, 0)
    if "lambda1" in s:
        ls.SimInspiralWaveformParamsInsertTidalLambda1(p, s["lambda1"])
        ls.SimInspiralWaveformParamsInsertTidalLambda2(p, s["lambda2"])
    spins = [s.get(f"spin{body}{axis}", 0.0) for body in (1, 2) for axis in "xyz"]
    hp, hc = ls.SimInspiralChooseFDWaveform(
        s["mass1"] * lal.MSUN_SI, s["mass2"] * lal.MSUN_SI, *spins, s["distance"] * 1e6 * lal.PC_SI,
        s["inclination"], s["coa_phase"], 0, 0, 0, delta_f, F_LOW, RATE / 2, F_LOW, p, getattr(ls, approximant))
    return hp.data.data, hc.data.data


def check(hp, hc, signals, approximant, delta_f, tolerance):
    for row, s in enumerate(signals):
        for ours, theirs in zip((hp[row].numpy(), hc[row].numpy()), lal_hphc(s, approximant, delta_f)):
            assert len(ours) == len(theirs)
            band = theirs != 0
            assert np.abs(ours[band] - theirs[band]).max() <= tolerance * np.abs(theirs).max(), (approximant, s)
            assert not ours[~band].any(), (approximant, s)  # zero where LAL is zero


def grid(batch):
    return (waveform_utils.get_freqs(F_LOW, RATE / 2, T, batch, "cpu", torch.float64),
            waveform_utils.reference_frequencies(F_LOW, batch, "cpu"))


@pytest.mark.parametrize("model, tolerance", [(IMRPhenomD, 1e-12), (IMRPhenomXAS, 1e-12)])
def test_aligned(model, tolerance):
    hp, hc = model(*grid(len(ALIGNED)), 2.0).get_hphc(theta_of(ALIGNED), reproduce_lal=True)
    check(hp, hc, ALIGNED, model.__name__, 1 / T, tolerance)


def test_xhm():
    hp, hc = IMRPhenomXHM(*grid(len(ALIGNED)), padding_length_in_s=2.0).get_hphc(theta_of(ALIGNED), reproduce_lal=True)
    check(hp, hc, ALIGNED, "IMRPhenomXHM", 1 / T, 1e-5)


def test_xphm():
    model = IMRPhenomXPHM.from_grid(*grid(len(PRECESSING)), padding_length_in_s=2.0)
    hp, hc = model.get_hphc(theta_of(PRECESSING), reproduce_lal=True)
    check(hp, hc, PRECESSING, "IMRPhenomXPHM", 1 / T, 1e-5)


def test_pv2():
    """The production training model, built as a run builds it."""
    window, detectors, dtype = build_data({"detectors": ["H1", "L1"], "sample_rate_in_hz": RATE,
                                           "sample_length_in_s": 12.0, "padding_length_in_s": 2.0, "dtype": "float32"})
    section = {"prior": "bbh_narrow", "low_frequency_cutoff_in_hz": F_LOW, "reference_frequency_in_hz": F_LOW,
               "waveform": {"name": "IMRPhenomPv2"}, "projection": {"name": "constant"},
               "snr": {"name": "halfnorm", "scale": "4.0", "loc": "5.0"}, "targets": ["tc", "mchirp"]}
    model = build_signal(section, load_prior("bbh_narrow", Path(".")), window, detectors,
                         torch.ones(2, window.n_frequency_bins), len(PRECESSING), 1, 0, "cpu", dtype)
    hp, hc = model.get_hphc(theta_of(PRECESSING), reproduce_lal=True)
    check(hp, hc, PRECESSING, "IMRPhenomPv2", 1 / T, 1e-12)


def test_nrtidalv3():
    model = IMRPhenomXAS_NRTidalv3.from_band(1 / 256, F_LOW, RATE / 2, F_LOW, len(BNS), "cpu")
    hp, hc = model.get_hphc(theta_of(BNS), reproduce_lal=True)
    check(hp, hc, BNS, "IMRPhenomXAS_NRTidalv3", 1 / 256, 1e-10)
