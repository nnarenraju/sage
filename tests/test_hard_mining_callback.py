"""Tests for HardMiningCallback's keep-threshold resolution.

The callback accepts the "how signal-like must noise look to be kept" bar as
either a raw detection logit (``keep_threshold_raw``) or a probability
(``keep_threshold_sigmoided``); raw overrides sigmoided, neither set keeps every
window (-inf). Only the resolution logic is exercised here — constructing the
callback needs no pyribs/GPU (the miner is built lazily), so this runs in CI.
"""

import math

import numpy as np
import pytest

from sage.factory.callbacks import HardMiningCallback

# The bank lives under bank_dir but is built lazily (first mine), so construction
# never touches it -- a dummy path is enough to exercise threshold resolution.
_BANK = "/tmp/_hmcb_test_bank"


def _cb(**kw):
    return HardMiningCallback(bank_dir=_BANK, **kw)


def test_sigmoided_converts_to_logit():
    cb = _cb(keep_threshold_sigmoided=0.88)
    assert math.isclose(cb.keep_threshold, math.log(0.88 / 0.12), rel_tol=1e-9)


def test_raw_used_directly():
    cb = _cb(keep_threshold_raw=2.0)
    assert cb.keep_threshold == 2.0


def test_raw_overrides_sigmoided():
    cb = _cb(keep_threshold_raw=3.0, keep_threshold_sigmoided=0.5)
    assert cb.keep_threshold == 3.0          # raw wins; not logit(0.5)=0


def test_default_keeps_everything():
    cb = _cb()
    assert cb.keep_threshold == float("-inf")


def test_sigmoided_half_is_zero_logit():
    cb = _cb(keep_threshold_sigmoided=0.5)
    assert math.isclose(cb.keep_threshold, 0.0, abs_tol=1e-12)


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.2, 2.0])
def test_invalid_probability_rejected(bad):
    with pytest.raises(ValueError):
        _cb(keep_threshold_sigmoided=bad)


def test_resolved_threshold_is_plain_float():
    cb = _cb(keep_threshold_sigmoided=0.7)
    assert isinstance(cb.keep_threshold, float)
    assert np.isfinite(cb.keep_threshold)


def test_attributes_retained():
    cb = _cb(keep_threshold_raw=1.5, keep_threshold_sigmoided=0.9)
    # both inputs are stored for introspection even though raw wins
    assert cb.keep_threshold_raw == 1.5
    assert cb.keep_threshold_sigmoided == 0.9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
