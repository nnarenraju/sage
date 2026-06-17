"""CPU regression test: RecolourPostprocess._gather must be batch-size-agnostic.

The hard-noise miner reads variable batch sizes (e.g. 16 from the CMA-ES emitter),
not the configured training batch. _gather previously allocated its output with
the fixed ``self.B``, crashing ("could not broadcast (16,F) into (128,F)") on any
other batch size. This pins it to the actual input batch.
"""

import numpy as np
import torch

from sage.data.noise.recolour import RecolourPostprocess


def _bare(D=2):
    r = RecolourPostprocess.__new__(RecolourPostprocess)   # skip ASD-file loading
    r.D = D
    return r


def test_gather_uses_actual_batch_size():
    r = _bare(D=2)
    banks = [np.random.rand(100, 50).astype(np.float32) for _ in range(2)]
    for B in (16, 128, 1):                       # any batch, not a fixed self.B
        idx = np.random.randint(0, 100, (B, 2))
        out = r._gather(banks, idx)
        assert out.shape == (B, 2, 50), (B, out.shape)
        assert isinstance(out, torch.Tensor)


if __name__ == "__main__":
    test_gather_uses_actual_batch_size()
    print("  PASS test_gather_uses_actual_batch_size")
    print(">>> RECOLOUR TESTS PASSED <<<")
