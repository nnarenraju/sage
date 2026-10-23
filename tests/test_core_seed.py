"""Unit tests for sage.core.seed — SeedManager reproducibility."""

import random
import numpy as np
from sage.core.seed import SeedManager


class TestSeedManager:


    def test_numpy_global_rng_seeded(self):
        SeedManager(0)
        a = np.random.rand(10)
        SeedManager(0)
        b = np.random.rand(10)
        assert np.array_equal(a, b)

    def test_python_random_seeded(self):
        SeedManager(123)
        a = [random.random() for _ in range(5)]
        SeedManager(123)
        b = [random.random() for _ in range(5)]
        assert a == b


    def test_rng_reproducible_across_instances(self):
        a = SeedManager(1).rng.random(10)
        b = SeedManager(1).rng.random(10)
        assert np.allclose(a, b)

    def test_rng_independent_of_global_numpy(self):
        sm = SeedManager(5)
        # Advance global state
        np.random.rand(1000)
        val_after = sm.rng.random()
        # Re-create and don't touch global state
        sm2 = SeedManager(5)
        val_before = sm2.rng.random()
        assert val_before == val_after

    def test_spawn_same_name_same_seed(self):
        sm = SeedManager(99)
        r1 = sm.spawn("noise").random()
        r2 = sm.spawn("noise").random()
        assert r1 == r2

    def test_spawn_different_names_differ(self):
        sm = SeedManager(99)
        r1 = sm.spawn("noise").random()
        r2 = sm.spawn("signal").random()
        assert r1 != r2


    def test_spawn_deterministic_across_instances(self):
        r1 = SeedManager(42).spawn("waveform").random(5)
        r2 = SeedManager(42).spawn("waveform").random(5)
        assert np.allclose(r1, r2)

    def test_two_children_independent(self):
        sm = SeedManager(7)
        noise_child = sm.spawn("noise")
        signal_child = sm.spawn("signal")
        # Advancing one child should not affect the other
        noise_child.random(100)
        val_signal = signal_child.random()
        sm2 = SeedManager(7)
        sm2.spawn("noise")  # same spawn but don't advance
        val_signal2 = sm2.spawn("signal").random()
        assert val_signal == val_signal2
