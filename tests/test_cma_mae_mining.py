"""CPU tests for the pyribs CMA-MAE hard-noise miner (no model, no GPU, no data).

A trivial multi-modal ``evaluate_fn`` (three well-separated "hard" bumps in
start-time space, with an embedding that encodes position) stands in for the
read-noise -> model -> embedding pipeline.  We check the pyribs ask/tell loop
runs end-to-end, that the emitted StartTimeDataset only holds above-threshold
windows, and that mining covers more than one bump (QD diversity, no collapse).
"""

import numpy as np
import pytest

pytest.importorskip("ribs", reason="pyribs required for the CMA-MAE miner")

from sage.data.noise.cma_mae_mining import CMAMAEMiner, _StartTimeCodec

NSAMP, NSEG, SEQ = 5000, 2, 100
TOTAL = NSEG * NSAMP


def _seg():
    dt = np.dtype([("idx", "i8"), ("start", "i8"), ("end", "i8"), ("nsamples", "i8")])
    arr = np.zeros(NSEG, dtype=dt)
    arr["idx"] = np.arange(NSEG)
    arr["start"] = np.arange(NSEG) * NSAMP
    arr["nsamples"] = NSAMP
    arr["end"] = arr["start"] + NSAMP
    return arr


def _make_eval(seed=0):
    rng = np.random.default_rng(seed)
    bumps = np.array([0.2, 0.5, 0.8])
    sigma = 0.05

    def evaluate_fn(starts, segs):
        p = starts[:, 0].astype(float) / TOTAL                       # position in [0,1)
        peaks = np.stack([np.exp(-((p - b) ** 2) / (2 * sigma ** 2)) for b in bumps])
        scores = peaks.max(0) * 10.0 + 0.01 * rng.standard_normal(len(p))
        emb = np.stack(
            [np.sin(2 * np.pi * p), np.cos(2 * np.pi * p), p, p * p,
             np.sin(4 * np.pi * p), np.cos(4 * np.pi * p)], axis=1,
        ) + 0.001 * rng.standard_normal((len(p), 6))
        return scores, emb

    return evaluate_fn, bumps


def _miner(**kw):
    defaults = dict(
        detectors=["H1", "L1"], seg_index=[_seg(), _seg()], seq_len=SEQ,
        bin_files=["/x/H1.bin", "/x/L1.bin"], sample_rate=2048.0,
        keep_threshold=5.0, descriptor_dim=4, n_cells=16, learning_rate=0.1,
        threshold_min=0.0, n_emitters=2, emitter_batch_size=16, sigma0=0.2,
        n_warmup=128, seed=0,
    )
    defaults.update(kw)
    return CMAMAEMiner(**defaults)


def test_codec_decode_encode_stable():
    codec = _StartTimeCodec([_seg(), _seg()], SEQ)
    g = np.random.default_rng(0).random((64, 2))
    s, sg = codec.decode(g)
    s2, sg2 = codec.decode(codec.encode(s, sg))      # decode∘encode∘decode is a fixpoint
    assert np.array_equal(s, s2) and np.array_equal(sg, sg2)


def test_decode_respects_segment_bounds():
    codec = _StartTimeCodec([_seg()], SEQ)
    g = np.linspace(0, 1, 200)[:, None]
    s, sg = codec.decode(g)
    # every window must fit fully inside its segment
    seg = _seg()
    for i in range(len(s)):
        base = seg["start"][sg[i, 0]]
        assert base <= s[i, 0] <= base + (NSAMP - SEQ)


def test_mine_returns_only_hard_windows():
    ev, _ = _make_eval(0)
    ds = _miner().mine(ev, n_iters=20)
    assert len(ds) > 0
    assert (ds.scores >= 5.0).all()
    assert ds.start_indices.shape[1] == 2 and ds.detectors == ["H1", "L1"]


def test_mine_is_diverse_not_collapsed():
    ev, bumps = _make_eval(0)
    ds = _miner().mine(ev, n_iters=30)
    p = ds.start_indices[:, 0].astype(float) / TOTAL
    hit = {int(np.argmin(np.abs(bumps - pi))) for pi in p if np.min(np.abs(bumps - pi)) < 0.06}
    assert len(hit) >= 2, f"QD collapsed onto bump(s) {hit}"
    print(f"  kept {len(ds)} hard windows covering bumps {sorted(hit)}")


def test_empty_when_nothing_passes_threshold():
    ev, _ = _make_eval(0)
    ds = _miner(keep_threshold=1e6).mine(ev, n_iters=5)   # impossibly high bar
    assert len(ds) == 0 and ds.start_indices.shape == (0, 2)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print(">>> ALL CMA-MAE MINER TESTS PASSED <<<")
