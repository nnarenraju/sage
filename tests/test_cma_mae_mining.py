"""CPU tests for the pyribs CMA-MAE hard-noise miner (no model, no GPU, no data).

A trivial multi-modal ``evaluate_fn`` (three well-separated "hard" bumps in
start-time space, with an embedding that encodes position) stands in for the
read-noise -> model -> embedding pipeline.  We check the pyribs ask/tell loop
runs end-to-end, that the file-backed bank only stores above-threshold windows,
that mining covers more than one bump (QD diversity, no collapse), and that the
(run, segment, start) codec round-trips including the run id (multi-run).
"""
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("ribs", reason="pyribs required for the CMA-MAE miner")

from sage.data.noise.cma_mae_mining import CMAMAEMiner, _StartTimeCodec
from sage.data.noise.hard_bank import HardMiningBank

NSAMP, NSEG, SEQ = 5000, 2, 100
TOTAL = NSEG * NSAMP
_SEG_DT = np.dtype([("run", "i8"), ("idx", "i8"), ("start", "i8"),
                    ("end", "i8"), ("nsamples", "i8")])


def _seg(run=0):
    """One detector's segment table (single run)."""
    arr = np.zeros(NSEG, dtype=_SEG_DT)
    arr["run"] = run
    arr["idx"] = np.arange(NSEG)
    arr["start"] = np.arange(NSEG) * NSAMP
    arr["nsamples"] = NSAMP
    arr["end"] = arr["start"] + NSAMP
    return arr


def _seg_pooled():
    """One detector's POOLED table over two runs (segment ids collide across
    runs; only (run, idx) is unique)."""
    return np.concatenate([_seg(run=0), _seg(run=1)])


def _make_eval(seed=0):
    rng = np.random.default_rng(seed)
    bumps = np.array([0.2, 0.5, 0.8])
    sigma = 0.05

    def evaluate_fn(starts, segs, runs):
        p = starts[:, 0].astype(float) / TOTAL                       # position in [0,1)
        peaks = np.stack([np.exp(-((p - b) ** 2) / (2 * sigma ** 2)) for b in bumps])
        scores = peaks.max(0) * 10.0 + 0.01 * rng.standard_normal(len(p))
        emb = np.stack(
            [np.sin(2 * np.pi * p), np.cos(2 * np.pi * p), p, p * p,
             np.sin(4 * np.pi * p), np.cos(4 * np.pi * p)], axis=1,
        ) + 0.001 * rng.standard_normal((len(p), 6))
        return scores, emb

    return evaluate_fn, bumps


def _bank(tmp, runs=("O3b",)):
    return HardMiningBank(
        os.path.join(tmp, "bank.h5"), detectors=["H1", "L1"], runs=list(runs),
        seq_len=SEQ, sample_rate=2048.0, bin_files=["/x/H1.bin", "/x/L1.bin"],
        descriptor_dim=4,
    )


def _miner(tmp, seg_index=None, **kw):
    defaults = dict(
        detectors=["H1", "L1"], seg_index=seg_index or [_seg(), _seg()], seq_len=SEQ,
        bank=_bank(tmp), keep_threshold=5.0, descriptor_dim=4, n_cells=16,
        learning_rate=0.1, threshold_min=0.0, n_emitters=2, emitter_batch_size=16,
        sigma0=0.2, n_warmup=128, seed=0,
    )
    defaults.update(kw)
    return CMAMAEMiner(**defaults)


# ── codec ──────────────────────────────────────────────────────────────────
def test_codec_decode_encode_stable():
    codec = _StartTimeCodec([_seg(), _seg()], SEQ)
    g = np.random.default_rng(0).random((64, 2))
    s, sg, rn = codec.decode(g)
    # decode∘encode∘decode is a fixpoint (carries run too)
    s2, sg2, rn2 = codec.decode(codec.encode(s, sg, rn))
    assert np.array_equal(s, s2) and np.array_equal(sg, sg2) and np.array_equal(rn, rn2)


def test_codec_roundtrips_run_for_pooled_runs():
    """A pooled two-run table: decode must report the correct run, and
    encode/decode must round-trip the (run, segment, start) identity."""
    codec = _StartTimeCodec([_seg_pooled(), _seg_pooled()], SEQ)
    g = np.random.default_rng(1).random((256, 2))
    s, sg, rn = codec.decode(g)
    assert set(np.unique(rn).tolist()) == {0, 1}, "decode never reaches run 1"
    s2, sg2, rn2 = codec.decode(codec.encode(s, sg, rn))
    assert np.array_equal(s, s2) and np.array_equal(sg, sg2) and np.array_equal(rn, rn2)


def test_decode_respects_segment_bounds():
    codec = _StartTimeCodec([_seg()], SEQ)
    g = np.linspace(0, 1, 200)[:, None]
    s, sg, rn = codec.decode(g)
    # every window must fit fully inside its segment
    seg = _seg()
    for i in range(len(s)):
        base = seg["start"][sg[i, 0]]
        assert base <= s[i, 0] <= base + (NSAMP - SEQ)


# ── mining (file-backed bank) ──────────────────────────────────────────────
def test_mine_keeps_only_hard_windows():
    with tempfile.TemporaryDirectory() as tmp:
        m = _miner(tmp)
        ev, _ = _make_eval(0)
        stats = m.mine(ev, n_iters=20, epoch=0)
        assert stats["kept_starts"] > 0 and m.bank.n_starts > 0
        # every banked window cleared the keep threshold
        _, _, _ = m.bank.read_starts(np.arange(m.bank.n_starts))
        import h5py
        with h5py.File(m.bank.path, "r") as f:
            assert (f["start_found_score"][:] >= 5.0).all()
            assert f["start_runs"].shape == f["start_times"].shape


def test_mine_is_diverse_not_collapsed():
    with tempfile.TemporaryDirectory() as tmp:
        m = _miner(tmp)
        ev, bumps = _make_eval(0)
        m.mine(ev, n_iters=30, epoch=0)
        s, _, _ = m.bank.read_starts(np.arange(m.bank.n_starts))
        p = s[:, 0].astype(float) / TOTAL
        hit = {int(np.argmin(np.abs(bumps - pi))) for pi in p
               if np.min(np.abs(bumps - pi)) < 0.06}
        assert len(hit) >= 2, f"QD collapsed onto bump(s) {hit}"


def test_mine_empty_when_nothing_passes_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        m = _miner(tmp, keep_threshold=1e6)          # impossibly high bar
        ev, _ = _make_eval(0)
        stats = m.mine(ev, n_iters=5, epoch=0)
        assert stats["kept_starts"] == 0 and m.bank.n_starts == 0
