"""Unit tests for sage.data.noise.hard_mining (CPU-only, no model required)."""

import pytest

tqdm = pytest.importorskip("tqdm", reason="tqdm not installed")

import torch
from sage.data.noise.hard_mining import HardSampleBuffer, HardSampleMiner


# ---------------------------------------------------------------------------
# HardSampleBuffer
# ---------------------------------------------------------------------------

class TestHardSampleBuffer:
    def test_initial_state_len_zero(self):
        buf = HardSampleBuffer(capacity=100)
        assert len(buf) == 0

    def test_initial_not_ready(self):
        buf = HardSampleBuffer(capacity=100)
        assert buf.is_ready is False

    def test_sample_before_replace_returns_none(self):
        buf = HardSampleBuffer(capacity=100)
        assert buf.sample(10, device="cpu") is None

    def test_top_logit_before_replace_is_nan(self):
        import math
        buf = HardSampleBuffer(capacity=100)
        assert math.isnan(buf.top_logit)

    def test_median_logit_before_replace_is_nan(self):
        import math
        buf = HardSampleBuffer(capacity=100)
        assert math.isnan(buf.median_logit)

    def test_replace_populates_buffer(self):
        buf = HardSampleBuffer(capacity=50)
        data = torch.randn(30, 8)
        logits = torch.randn(30)
        buf.replace(data, logits)
        assert buf.is_ready is True
        assert len(buf) == 30

    def test_replace_respects_capacity(self):
        capacity = 20
        buf = HardSampleBuffer(capacity=capacity)
        data = torch.randn(100, 4)
        logits = torch.randn(100)
        buf.replace(data, logits)
        assert len(buf) == capacity

    def test_replace_keeps_top_logits(self):
        buf = HardSampleBuffer(capacity=5)
        data = torch.randn(10, 2)
        logits = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.5, 0.2, 0.7, 0.4, 0.6, 1.0])
        buf.replace(data, logits)
        # After keeping top-5: logits should be [1.0, 0.9, 0.8, 0.7, 0.6] (sorted desc)
        assert len(buf) == 5
        retained = buf._logits
        # All retained logits must be >= the minimum of top-5
        top5 = torch.topk(logits, 5).values
        assert (retained >= top5.min() - 1e-6).all()

    def test_top_logit_is_maximum(self):
        buf = HardSampleBuffer(capacity=100)
        data = torch.randn(20, 4)
        logits = torch.randn(20)
        buf.replace(data, logits)
        # top_logit should be the largest logit in the buffer
        assert buf.top_logit == pytest.approx(buf._logits.max().item(), rel=1e-6)

    def test_median_logit_matches_torch_median(self):
        buf = HardSampleBuffer(capacity=100)
        data = torch.randn(30, 4)
        logits = torch.arange(30, dtype=torch.float32)
        buf.replace(data, logits)
        assert buf.median_logit == pytest.approx(buf._logits.median().item(), rel=1e-6)

    def test_sample_returns_correct_shape(self):
        buf = HardSampleBuffer(capacity=100)
        C = 8
        data = torch.randn(50, C)
        logits = torch.randn(50)
        buf.replace(data, logits)
        n = 10
        samples = buf.sample(n, device="cpu")
        assert samples is not None
        assert samples.shape == (n, C)

    def test_sample_clamps_to_buffer_size(self):
        buf = HardSampleBuffer(capacity=10)
        data = torch.randn(5, 4)
        logits = torch.randn(5)
        buf.replace(data, logits)
        # Requesting more than available should return at most len(buf) items
        samples = buf.sample(100, device="cpu")
        assert samples.shape[0] <= len(buf)

    def test_sample_on_cpu_device(self):
        buf = HardSampleBuffer(capacity=50)
        data = torch.randn(20, 6)
        logits = torch.randn(20)
        buf.replace(data, logits)
        samples = buf.sample(5, device="cpu", dtype=torch.float32)
        assert samples.device.type == "cpu"

    def test_replace_updates_buffer(self):
        buf = HardSampleBuffer(capacity=100)
        data1 = torch.randn(10, 4)
        logits1 = torch.zeros(10)
        buf.replace(data1, logits1)
        first_top = buf.top_logit

        data2 = torch.randn(10, 4)
        logits2 = torch.ones(10) * 5.0  # much higher
        buf.replace(data2, logits2)
        assert buf.top_logit > first_top


# ---------------------------------------------------------------------------
# HardSampleMiner._streaming_top_k (static method)
# ---------------------------------------------------------------------------

class TestStreamingTopK:
    def test_prunes_to_capacity(self):
        capacity = 5
        acc_data, acc_logits = [], []
        data = torch.randn(20, 4)
        logits = torch.randn(20)
        HardSampleMiner._streaming_top_k(data, logits, acc_data, acc_logits, capacity)
        total = torch.cat(acc_logits).shape[0]
        assert total == capacity

    def test_retains_top_logits(self):
        capacity = 3
        acc_data, acc_logits = [], []
        logits = torch.tensor([0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 1.0])
        data = torch.randn(len(logits), 2)
        HardSampleMiner._streaming_top_k(data, logits, acc_data, acc_logits, capacity)
        retained = torch.cat(acc_logits)
        top3 = torch.topk(logits, capacity).values.min()
        assert (retained >= top3 - 1e-6).all()

    def test_appends_when_below_capacity(self):
        capacity = 100
        acc_data, acc_logits = [], []
        data = torch.randn(10, 4)
        logits = torch.randn(10)
        HardSampleMiner._streaming_top_k(data, logits, acc_data, acc_logits, capacity)
        # Below capacity — just appended, no pruning
        total = torch.cat(acc_logits).shape[0]
        assert total == 10

    def test_multiple_calls_accumulate_and_prune(self):
        capacity = 5
        acc_data, acc_logits = [], []
        for _ in range(4):
            data = torch.randn(10, 2)
            logits = torch.rand(10)
            HardSampleMiner._streaming_top_k(
                data, logits, acc_data, acc_logits, capacity
            )
        total = torch.cat(acc_logits).shape[0]
        assert total <= capacity
