"""
Production smoke test for the *base* o3b training pipeline (no hard-mining).

Unlike the unit tests, this validates the REAL run in ``runs/o3b/`` against the
REAL data release: it loads ``runs/o3b/config.py`` + ``train.py``, builds the
exact training / validation graphs and model used in production, and runs a
handful of train + validation iterations end to end, asserting the loss is
finite. The intent is to catch "production run is mis-wired" issues (bad paths,
missing fiducial PSDs, shape/device mismatches) before launching a full run.

It needs CUDA, the data release under ``/data/wiay/...`` and the rebuilt
fiducial PSDs, so it is automatically skipped where that environment is absent
(e.g. CI). Runs in eager mode — ``torch.compile`` autotune is covered separately
(see test_compiled.py).
"""

import os
import sys
import time
import importlib
from contextlib import nullcontext
from pathlib import Path

import torch

try:
    import pytest
except ModuleNotFoundError:
    # Allow standalone execution under the sage conda python (which lacks
    # pytest) — the decorators below degrade to no-ops and the __main__ block
    # drives the tests directly. Under real pytest this branch is skipped.
    class _MarkStub:
        def __getattr__(self, _name):
            def _decorator(*args, **kwargs):
                if len(args) == 1 and callable(args[0]) and not kwargs:
                    return args[0]
                return lambda fn: fn
            return _decorator

    class _PytestStub:
        mark = _MarkStub()

        @staticmethod
        def fixture(*args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            return lambda fn: fn

    pytest = _PytestStub()

# ── Production run location + data release ────────────────────────────────
RUN_DIR = Path(__file__).resolve().parent.parent / "runs" / "o3b"
DATA_RELEASE = Path("/data/wiay/nnarenraju/data_release")

N_TRAIN_ITERS = 3
N_VAL_ITERS = 2

# ── Environment gates: only run where the real run can actually execute ───
_HAS_CUDA = torch.cuda.is_available()
_HAS_NOISE = (DATA_RELEASE / "o3b_dataset" / "data_H1_O3b.bin").exists()
_HAS_FIDUCIAL = (RUN_DIR / "run_export" / "fiducial_psds" / "fiducial_H1_psd.bin").exists()

requires_run_env = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_NOISE and _HAS_FIDUCIAL),
    reason=(
        f"needs CUDA ({_HAS_CUDA}), data release ({_HAS_NOISE}) and "
        f"rebuilt fiducial PSDs ({_HAS_FIDUCIAL})"
    ),
)


def _stage(name):
    print(f"\n{'='*68}\n[STAGE] {name}\n{'='*68}", flush=True)


@pytest.fixture(scope="module")
def o3b_run():
    """Import the production runs/o3b config+train with the run dir as cwd.

    The run uses relative paths (``./gwconfig.yaml``, ``./run_export``), so we
    must chdir into runs/o3b and put it on sys.path, then register configs.
    """
    prev_cwd = os.getcwd()
    sys.path.insert(0, str(RUN_DIR))
    os.chdir(RUN_DIR)
    for mod in ("config", "train"):
        sys.modules.pop(mod, None)
    config = importlib.import_module("config")
    train = importlib.import_module("train")
    config.set_configs()

    yield config, train

    os.chdir(prev_cwd)
    if str(RUN_DIR) in sys.path:
        sys.path.remove(str(RUN_DIR))
    for mod in ("config", "train"):
        sys.modules.pop(mod, None)


@requires_run_env
def test_o3b_production_config_sane(o3b_run):
    """The production config points at real, present data + fiducial PSDs."""
    _stage("Validate production config + data files")
    from sage.core.config import get_cfg, get_data_cfg

    cfg, data_cfg = get_cfg(), get_data_cfg()
    print(f"  device={cfg.device} batch_size={cfg.batch_size} detectors={cfg.detectors}", flush=True)

    assert cfg.detectors == ["H1", "L1"]
    assert data_cfg.sample_rate == 2048.0

    for f in list(data_cfg.training_noise_files) + list(data_cfg.validation_noise_files):
        assert Path(f).exists(), f"missing noise file: {f}"

    for det in cfg.detectors:
        fid = RUN_DIR / "run_export" / "fiducial_psds" / f"fiducial_{det}_psd.bin"
        assert fid.exists(), f"missing fiducial PSD: {fid}"
    print("  config + data files OK", flush=True)


@requires_run_env
def test_o3b_pipeline_runs_with_finite_loss(o3b_run):
    """End-to-end: build production graphs+model, run a few iters, loss finite."""
    config, train = o3b_run
    from sage.core.config import get_cfg, get_data_cfg

    cfg, data_cfg = get_cfg(), get_data_cfg()

    # Start from a clean validation file so repeated test runs don't accumulate
    # or collide on epoch groups.
    vfile = RUN_DIR / "run_export" / "validation_data.h5"
    if vfile.exists():
        vfile.unlink()

    _stage("1. Build TRAINING graph (IMRPhenomPv2 signal + recoloured o3b noise)")
    train_signal, train_noise, bounds = train.make_training_graph()
    print("  training graph built", flush=True)

    _stage("2. Build VALIDATION graph (validation noise = o3a)")
    val_signal, val_noise = train.make_validation_graph()
    print("  validation graph built", flush=True)

    _stage("3. Build processor (FiducialWhitening + MultirateSampler)")
    processor = train.make_processor(bounds)
    print("  processor built", flush=True)

    _stage("4. Build model (MSCNN1D_2DResNetCBAM_Heteroscedastic, eager)")
    from sage.architecture.network import MSCNN1D_2DResNetCBAM_Heteroscedastic

    model = MSCNN1D_2DResNetCBAM_Heteroscedastic(
        frontend_filters=32,
        frontend_kernel=64,
        backend_resnet_size=50,
        norm_type="instancenorm",
    ).to(dtype=cfg.dtype, device=cfg.device, memory_format=torch.channels_last)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    _stage("5. Loss / optimiser / scheduler / scaler")
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    from sage.architecture.custom_losses import BCEWithPEsigmaLoss

    loss_function = BCEWithPEsigmaLoss(regression_weight=0.005, coupling_weight=0.005)
    optimiser = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6, fused=True)
    scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler(cfg.device, enabled=cfg.autocast)

    _stage(f"6. TRAINING — {N_TRAIN_ITERS} iterations")
    from sage.factory.training import SageVanillaTraining

    train_sage = SageVanillaTraining(
        train_signal, train_noise, processor, model, loss_function,
        optimiser, scheduler, scaler,
        num_iterations=N_TRAIN_ITERS, num_epochs=1,
    )
    train_sage(nepoch=0)
    train_loss = train_sage.loss_components[0]
    print(f"  train loss_components[0] = {train_loss.tolist()}", flush=True)
    assert torch.isfinite(train_loss).all(), f"non-finite train loss: {train_loss}"

    _stage(f"7. VALIDATION — {N_VAL_ITERS} iterations (writes run_export/validation_data.h5)")
    from sage.factory.validation import SageVanillaValidation

    validate_sage = SageVanillaValidation(
        val_signal, val_noise, processor, model, loss_function,
        num_iterations=N_VAL_ITERS, num_epochs=1,
    )
    validate_sage(nepoch=0)
    val_loss = validate_sage.loss_components[0]
    print(f"  val loss_components[0] = {val_loss.tolist()}", flush=True)
    assert torch.isfinite(val_loss).all(), f"non-finite val loss: {val_loss}"

    _stage("o3b PRODUCTION SMOKE TEST PASSED")


# ── Longer, compiled-model run + per-stage shape probe ─────────────────────
N_LONG_TRAIN_ITERS = 40
N_LONG_VAL_ITERS = 8


def _shp(name, t):
    if isinstance(t, (tuple, list)):
        for i, e in enumerate(t):
            _shp(f"{name}[{i}]", e)
    elif hasattr(t, "shape"):
        print(f"    {name:26s} shape={tuple(t.shape)}  dtype={t.dtype}  dev={t.device}")
    else:
        print(f"    {name:26s} {type(t).__name__}={t}")


def _probe_pipeline_shapes(cfg, signal_sampler, noise_sampler, processor, model, loss_function):
    """Run one training step stage by stage, printing the tensor shape/dtype
    at each: data gen -> combine -> transform -> net -> loss."""
    from sage.core.pipeline import GWBatch, Grid, ProcessingState

    B = cfg.batch_size
    S = int(B * cfg.class_balance)
    num_pe = len(cfg.do_point_estimate)
    num_targets = num_pe + 1
    device = cfg.device

    with torch.no_grad():
        _stage("SHAPE PROBE — one training step, stage by stage")

        print("  [1 data-gen] signal sampler (IMRPhenomPv2 + projection + SNR rescale):")
        sig, sig_t = signal_sampler()
        _shp("signal_data", sig)
        _shp("signal_targets", sig_t)

        print("  [1 data-gen] noise sampler (memmap O3b noise -> FD recolour):")
        noise, noise_t = noise_sampler()
        _shp("noise_data", noise)
        _shp("noise_targets", noise_t)

        selector = getattr(signal_sampler, "selector", None)
        print(f"  [2 multiband] selector = {selector} (None => FD_UNIFORM, no coarsening)")
        if selector is not None:
            noise = selector(noise)
            _shp("noise_data(coarse)", noise)

        print(f"  [3 combine] inject {S}/{B} signals (class_balance={cfg.class_balance}):")
        pad = torch.zeros(noise_t.shape[0], num_pe, device=device, dtype=noise_t.dtype)
        noise_t = torch.cat((pad, noise_t), dim=1)
        idx = torch.randperm(B, device=device)[:S]
        signal_pad = torch.zeros_like(noise)
        target_pad = torch.zeros(B, num_targets, device=device, dtype=sig_t.dtype)
        signal_pad[idx] = sig
        target_pad[idx] = sig_t
        x = noise + signal_pad
        targets = noise_t + target_pad
        _shp("x (noise+signal)", x)
        _shp("targets", targets)

        print("  [4 transform] GWBatch -> processor (FiducialWhitening + MultirateSampler):")
        initial_state = getattr(signal_sampler, "output_state", ProcessingState(Grid.FD_UNIFORM))
        freqs = selector.coarse_freqs if selector is not None else None
        coarse = selector.coarse_indices if selector is not None else None
        batch = GWBatch(x, state=initial_state, freqs=freqs, coarse_indices=coarse)
        print(f"    initial state : {initial_state}")
        batch = processor(batch)
        print(f"    post-processor: {getattr(batch, 'state', '?')}")
        net_input = batch.to_network_input()
        _shp("net_input", net_input)

        print("  [5 net] compiled model forward (autocast fp16):")
        with (torch.autocast(device_type="cuda", dtype=torch.float16)
              if cfg.autocast else nullcontext()):
            out = model(net_input)
            _shp("model_output", out)
            print("  [6 loss] loss_function(out, targets):")
            loss = loss_function(out, targets)
        _shp("loss", loss)
        print(f"    num_components={loss_function.num_components}  values={loss.detach().tolist()}")


def _profile_steps(cfg, signal_sampler, noise_sampler, processor, model,
                   loss_function, optimiser, scaler, n_iters=20):
    """Per-component per-iter timing: signal / noise / combine / processor /
    model fwd+bwd. cuda.synchronize between stages serialises the (normally
    overlapped) prefetch, so the printed total is an upper bound — read it for
    the *relative* split that localises any bottleneck."""
    from sage.core.pipeline import GWBatch, Grid, ProcessingState

    B = cfg.batch_size
    S = int(B * cfg.class_balance)
    npe = len(cfg.do_point_estimate)
    nt = npe + 1
    dev = cfg.device
    init_state = getattr(signal_sampler, "output_state", ProcessingState(Grid.FD_UNIFORM))
    sync = torch.cuda.synchronize
    agg = dict(signal=0.0, noise=0.0, combine=0.0, processor=0.0, model=0.0, total=0.0)

    for _ in range(n_iters):
        sync(); a = time.time()
        sd, st = signal_sampler(); sync(); b = time.time()
        nd, ntg = noise_sampler(); sync(); c = time.time()
        ntg = torch.cat((torch.zeros(ntg.shape[0], npe, device=dev, dtype=ntg.dtype), ntg), 1)
        idx = torch.randperm(B, device=dev)[:S]
        sp = torch.zeros_like(nd)
        tp = torch.zeros(B, nt, device=dev, dtype=st.dtype)
        sp[idx] = sd; tp[idx] = st
        x = nd + sp; tg = ntg + tp; sync(); d = time.time()
        batch = GWBatch(x, state=init_state, freqs=None, coarse_indices=None)
        batch = processor(batch); ni = batch.to_network_input(); sync(); e = time.time()
        optimiser.zero_grad(set_to_none=True)
        with (torch.autocast(device_type="cuda", dtype=torch.float16)
              if cfg.autocast else nullcontext()):
            out = model(ni); loss = loss_function(out, tg)
        scaler.scale(loss[0]).backward(); scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        scaler.step(optimiser); scaler.update(); sync(); f = time.time()
        agg['signal'] += b - a; agg['noise'] += c - b; agg['combine'] += d - c
        agg['processor'] += e - d; agg['model'] += f - e; agg['total'] += f - a

    print(f"  per-iter breakdown over {n_iters} iters (serialised; total is an upper bound):")
    for k in ("signal", "noise", "combine", "processor", "model", "total"):
        print(f"    {k:10s} {agg[k]/n_iters*1000:7.1f} ms  ({100*agg[k]/agg['total']:.0f}%)")


@requires_run_env
def test_o3b_pipeline_compiled_longrun(o3b_run):
    """Production-faithful longer run: COMPILED model (fullgraph + dynamic),
    a per-stage shape probe, then N steady-state iters. Everything except the
    model is eager. Kept O(minutes) by skipping max-autotune (default mode)."""
    config, train = o3b_run
    from sage.core.config import get_cfg, get_data_cfg

    cfg, data_cfg = get_cfg(), get_data_cfg()

    vfile = RUN_DIR / "run_export" / "validation_data.h5"
    if vfile.exists():
        vfile.unlink()

    _stage("Build production graphs (eager) + model")
    train_signal, train_noise, bounds = train.make_training_graph()
    val_signal, val_noise = train.make_validation_graph()
    processor = train.make_processor(bounds)

    from sage.architecture.network import MSCNN1D_2DResNetCBAM_Heteroscedastic

    model = MSCNN1D_2DResNetCBAM_Heteroscedastic(
        frontend_filters=32,
        frontend_kernel=64,
        backend_resnet_size=50,
        norm_type="instancenorm",
    ).to(dtype=cfg.dtype, device=cfg.device, memory_format=torch.channels_last)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    # Compile ONLY the model (production compiles the model; everything else is
    # eager). fullgraph=True asserts there are no graph breaks. Default mode
    # (not max-autotune) keeps compile O(minutes); the graph is the same.
    _stage("Compile model: torch.compile(fullgraph=True, dynamic=True), default mode")
    t = time.time()
    model = torch.compile(model, fullgraph=True, dynamic=True)
    print(f"  wrapper set ({time.time()-t:.2f}s); graph compiles on first forward")

    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    from sage.architecture.custom_losses import BCEWithPEsigmaLoss

    loss_function = BCEWithPEsigmaLoss(regression_weight=0.005, coupling_weight=0.005)
    optimiser = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6, fused=True)
    scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler(cfg.device, enabled=cfg.autocast)

    # Shape probe — first model() call here triggers the compile.
    t = time.time()
    _probe_pipeline_shapes(cfg, train_signal, train_noise, processor, model, loss_function)
    print(f"  (first compiled forward + probe took {time.time()-t:.1f}s incl. compile)")

    from sage.factory.training import SageVanillaTraining
    from sage.factory.validation import SageVanillaValidation

    _stage(f"LONG TRAINING — {N_LONG_TRAIN_ITERS} compiled iters (steady-state timing)")
    train_sage = SageVanillaTraining(
        train_signal, train_noise, processor, model, loss_function,
        optimiser, scheduler, scaler,
        num_iterations=N_LONG_TRAIN_ITERS, num_epochs=1,
    )
    t = time.time()
    train_sage(nepoch=0)
    dt = time.time() - t
    tl = train_sage.loss_components[0]
    print(f"  {N_LONG_TRAIN_ITERS} iters in {dt:.1f}s = {N_LONG_TRAIN_ITERS/dt:.2f} it/s "
          f"({cfg.batch_size*N_LONG_TRAIN_ITERS/dt:.0f} samples/s)")
    print(f"  train loss = {tl.tolist()}")
    assert torch.isfinite(tl).all(), f"non-finite train loss: {tl}"

    _stage("PER-COMPONENT PROFILE (localise where each iter's time goes)")
    _profile_steps(cfg, train_signal, train_noise, processor, model,
                   loss_function, optimiser, scaler, n_iters=20)

    _stage(f"LONG VALIDATION — {N_LONG_VAL_ITERS} iters (writes validation_data.h5)")
    validate_sage = SageVanillaValidation(
        val_signal, val_noise, processor, model, loss_function,
        num_iterations=N_LONG_VAL_ITERS, num_epochs=1,
    )
    validate_sage(nepoch=0)
    vl = validate_sage.loss_components[0]
    print(f"  val loss = {vl.tolist()}")
    assert torch.isfinite(vl).all(), f"non-finite val loss: {vl}"

    _stage("o3b COMPILED LONG-RUN TEST PASSED")


if __name__ == "__main__":
    # Also runnable standalone with the sage conda python (no pytest needed):
    #   python3 tests/test_o3b_pipeline.py
    if not (_HAS_CUDA and _HAS_NOISE and _HAS_FIDUCIAL):
        print(f"SKIP: needs CUDA={_HAS_CUDA}, data={_HAS_NOISE}, fiducial={_HAS_FIDUCIAL}")
        raise SystemExit(0)

    _prev = os.getcwd()
    sys.path.insert(0, str(RUN_DIR))
    os.chdir(RUN_DIR)
    for _mod in ("config", "train"):
        sys.modules.pop(_mod, None)
    import config as _config
    import train as _train
    _config.set_configs()
    _run = (_config, _train)
    try:
        if "--long" in sys.argv:
            # Compiled model + shape probe + steady-state throughput.
            test_o3b_pipeline_compiled_longrun(_run)
        else:
            test_o3b_production_config_sane(_run)
            test_o3b_pipeline_runs_with_finite_loss(_run)
        print("\n>>> ALL o3b PIPELINE CHECKS PASSED <<<", flush=True)
    finally:
        os.chdir(_prev)
