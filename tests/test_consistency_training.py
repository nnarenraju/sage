"""
Smoke test for the consistency training integration (eager, a few iterations).

Builds the real o3b consistency graph (signal sampler with per-detector tc,
recoloured noise, multirate processor), the consistency model + the combined
loss, and runs a handful of SageConsistencyTraining iterations, asserting the
loss stays finite. Skipped where CUDA / data / fiducial PSDs are absent.
"""

import os
import sys
import importlib
from pathlib import Path

import torch

RUN_DIR = Path(__file__).resolve().parent.parent / "runs" / "o3b"
DATA = Path("/local/scratch/igr/nnarenraju/data_release")

_HAS_CUDA = torch.cuda.is_available()
_HAS_NOISE = (DATA / "o3b_dataset" / "data_H1_O3b.bin").exists()
_HAS_FID = (RUN_DIR / "run_export" / "fiducial_psds" / "fiducial_H1_psd.bin").exists()
_READY = _HAS_CUDA and _HAS_NOISE and _HAS_FID

N_ITERS = 3


def _run():
    prev = os.getcwd()
    sys.path.insert(0, str(RUN_DIR))
    os.chdir(RUN_DIR)
    for m in ("config", "train_consistency"):
        sys.modules.pop(m, None)
    import config
    import train_consistency as tc
    config.O3bCFG.p_non_astrophysical = 0.3  # exercise the 4-class path
    config.set_configs()

    from sage.core.config import get_cfg, get_data_cfg
    from sage.architecture.network import MSCNN1D_2DResNetCBAM_Consistency, ConsistencyOutput
    from sage.architecture.custom_losses import BCEWithPEsigmaLoss, ConsistencyNLLLoss
    from sage.factory import SageConsistencyTraining
    from sage.data.non_astrophysical import NonAstrophysicalMasker
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    cfg, data_cfg = get_cfg(), get_data_cfg()
    signal_sampler, noise_sampler, bounds = tc.make_training_graph()
    processor, t_grid = tc.make_processor(bounds)

    # eager model (skip the multi-minute compile for a smoke test)
    model = MSCNN1D_2DResNetCBAM_Consistency(t_grid, dropout=0.05).to(
        dtype=cfg.dtype, device=cfg.device, memory_format=torch.channels_last
    )
    merged = BCEWithPEsigmaLoss(regression_weight=0.005, coupling_weight=0.005)
    cons = ConsistencyNLLLoss()
    opt = optim.Adam(model.parameters(), lr=2e-4, fused=True)
    sched = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler(cfg.device, enabled=cfg.autocast)

    masker = NonAstrophysicalMasker(
        freqs=signal_sampler.f[0],
        tc_bounds=bounds["tc"],
        analysis_length_s=data_cfg.sample_length_in_s,
        seed=1,
    )
    train_sage = SageConsistencyTraining(
        signal_sampler, noise_sampler, processor, model, merged, cons,
        opt, sched, scaler, num_iterations=N_ITERS, num_epochs=1,
        consistency_weight=0.1, masker=masker,
    )
    train_sage(nepoch=0)
    comps = train_sage.loss_components[0]

    os.chdir(prev)
    if str(RUN_DIR) in sys.path:
        sys.path.remove(str(RUN_DIR))
    return comps


def test_consistency_training_runs_with_finite_loss():
    if not _READY:
        print(f"SKIP: CUDA={_HAS_CUDA} noise={_HAS_NOISE} fiducial={_HAS_FID}")
        return
    comps = _run()
    print(f"  loss components [total, merged, consistency] = {comps.tolist()}")
    assert torch.isfinite(comps).all(), f"non-finite consistency loss: {comps}"


if __name__ == "__main__":
    test_consistency_training_runs_with_finite_loss()
    print(">>> CONSISTENCY TRAINING SMOKE TEST PASSED <<<")
