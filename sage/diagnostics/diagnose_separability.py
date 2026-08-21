#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : diagnose_separability.py
Description   : Prove the frontend is separable, on the real network and real strain.

Created on 2026-08-21

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The frontend cache re-uses one detector's features across every slide that re-pairs it.
That is valid only if a detector's features depend on that detector alone. The property
decides the cost of the whole programme -- measured on the O3b production network, the
cache is worth 2.7x, which over thirteen searches is the difference between seven days of
six GPUs and eighteen.

``norm_type: instancenorm`` in the checkpoint is a claim about the *training
configuration*. It is not a proof about the trained graph, and it is exactly the claim an
architecture refactor falsifies without changing the string. So the property is measured
on the weights: perturb one detector, and require every other detector's frontend output
to be **bitwise** unchanged.

Run on real strain rather than noise. A constant or synthetic input has no structure for a
coupling to act on, and a variance-normalising layer maps a constant to a degenerate
output dominated by its epsilon -- a probe that would pass on a coupled network.

The negative controls are not decoration. A gate that cannot fail proves nothing, so a
coupled frontend is built and must be refused before the real result is reported.

Usage
-----
    python -m sage.diagnostics.diagnose_separability --config config_o3a_HL
"""

import argparse
from pathlib import Path
from typing import Optional


def real_strain(spec, n_windows: int):
    """
    A batch of real, dyn-range-corrected strain windows from the campaign's own release.

    The first analysed windows rather than a random selection: the point is structured
    detector noise, and any of it will do; picking at random would make the report
    irreproducible for no gain.
    """
    import numpy as np

    from sage.search.grid import AnalysisGrid
    from sage.search.reader import StreamingStrainReader
    from sage.search.segments import coincident_intervals, load_segments

    geometry = spec.geometry_object()
    release = Path(spec.data.release_dir)
    segments = {
        detector: load_segments(
            release / f"data_{detector}_{spec.data.observing_run}_segments.json"
        )
        for detector in spec.data.detectors
    }
    grid = AnalysisGrid.build(
        geometry,
        segments,
        coincident_intervals(segments),
        reference_detector=spec.slides.reference_detector,
        coverage=False,
    )
    reader = StreamingStrainReader(
        release, grid, geometry, batch_size=max(n_windows, 8), prefetch=0
    )
    try:
        batch = next(iter(reader))
        strain = np.ascontiguousarray(np.asarray(batch.strain)[:n_windows])
        gps = float(batch.gps[0])
    finally:
        reader.close()
    return strain, gps


def perturbation_report(engine, strain) -> dict:
    """
    Perturb each detector in turn and measure every other detector's frontend change.

    The perturbation is a **substitution** -- the detector's windows are replaced by
    different real windows -- because that is what a time slide does. An additive offset
    would leave any shift-invariant coupling undetected, which is the failure mode that
    matters here: a frontend normalising by a statistic shared across detectors is
    coupled, and a constant offset moves that statistic not at all.

    Returns the worst cross-detector change and the worst self change; the second is what
    says the probe reached the network at all.
    """
    import numpy as np

    n_detectors = int(strain.shape[1])
    base = [engine.forward_frontend(strain, i) for i in range(n_detectors)]

    def features(value):
        return value[0] if isinstance(value, (tuple, list)) else value

    pairs = {}
    worst_cross = 0.0
    worst_self = 0.0
    for perturbed in range(n_detectors):
        altered = strain.copy()
        altered[:, perturbed, :] = strain[::-1, perturbed, :]
        after = [engine.forward_frontend(altered, i) for i in range(n_detectors)]
        for other in range(n_detectors):
            gap = float(
                (features(base[other]).float() - features(after[other]).float())
                .abs()
                .max()
            )
            pairs[(perturbed, other)] = gap
            if other == perturbed:
                worst_self = max(worst_self, gap)
            else:
                worst_cross = max(worst_cross, gap)
    return {
        "pairs": pairs,
        "worst_cross": worst_cross,
        "worst_self": worst_self,
        "separable": worst_cross == 0.0,
        "n_detectors": n_detectors,
    }


def controls(device: str) -> dict:
    """
    Check that the gate can fail, and that it does not fail on everything.

    ``groupnorm`` couples through a normalisation group spanning the detector axis;
    ``sharedscale`` couples through a shift-invariant statistic, which is the case an
    additive probe misses. Both must be refused. ``instancenorm`` must be accepted, or the
    gate rejects every network and its verdict on the real one means nothing.
    """
    import sys

    from sage.search.checkpoint import assert_separable

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tests.search_fixtures import ToyFrontendNet

    out = {}
    for norm, expected in (
        ("groupnorm", False),
        ("sharedscale", False),
        ("instancenorm", True),
    ):
        model = ToyFrontendNet(num_detectors=2, norm_type=norm).to(device)
        try:
            assert_separable(model)
            accepted = True
        except ValueError:
            accepted = False
        out[norm] = {"accepted": accepted, "correct": accepted == expected}
    return out


def main(argv: Optional[list] = None) -> int:
    """Run the controls, then the real measurement, and report a verdict."""
    parser = argparse.ArgumentParser(
        description="Prove the frontend cache is valid for this trained network."
    )
    parser.add_argument("--config", required=True, help="Campaign config module or path.")
    parser.add_argument("--windows", type=int, default=16, help="Windows to probe with.")
    parser.add_argument(
        "--device",
        default=None,
        help="Override the campaign's device. The verdict that counts is the one on the "
        "device the campaign runs on.",
    )
    args = parser.parse_args(argv)

    import dataclasses

    import torch

    from sage.search.checkpoint import as_config, assert_separable, load_search_model
    from sage.search.engine import SearchEngine, build_param_sampler, build_processor
    from sage.search.spec import load_spec

    spec = load_spec(args.config)
    device = args.device or (
        spec.engine.device if torch.cuda.is_available() else "cpu"
    )
    spec = dataclasses.replace(
        spec, engine=dataclasses.replace(spec.engine, device=device)
    )

    print(f"campaign {spec.tag}   device {device}")
    print(f"checkpoint {spec.engine.checkpoint}")

    verdicts = controls(device)
    print("\ncontrols:")
    for norm, result in verdicts.items():
        state = "accepted" if result["accepted"] else "refused"
        print(f"  {norm:14s} {state:9s} {'ok' if result['correct'] else 'WRONG'}")
    if not all(r["correct"] for r in verdicts.values()):
        print("\nthe gate does not behave on known cases; its verdict on the real "
              "network means nothing")
        return 1

    model, ckpt = load_search_model(
        spec.engine.checkpoint, cfg=None, data_cfg=None, device=device,
        architecture=spec.engine.architecture,
    )
    cfg, data_cfg = as_config(ckpt.cfg), as_config(ckpt.data_cfg)
    spec.apply_shadow_overrides(cfg, data_cfg)
    print(f"\nnorm_type recorded in the checkpoint: {ckpt.cfg.get('norm_type')!r} "
          "(a claim, not the measurement)")

    sampler = build_param_sampler(
        cfg, data_cfg, spec.engine.gwconfig, seed=int(spec.engine.sampler_seed)
    )
    engine = SearchEngine(
        model, build_processor(sampler), spec.geometry_object(), device=device,
        amp_dtype=spec.engine.amp_dtype,
        autocast=bool(ckpt.cfg.get("autocast", True)),
    )

    strain, gps = real_strain(spec, args.windows)
    print(f"real strain {strain.shape} from GPS {gps:.3f}")

    assert_separable(model, sample_input=torch.as_tensor(strain, device=device))
    print("assert_separable: PASS")

    report = perturbation_report(engine, strain)
    print(f"\nperturbation, bitwise, {report['n_detectors']} detectors:")
    for (perturbed, other), gap in sorted(report["pairs"].items()):
        kind = "self" if perturbed == other else "cross"
        print(f"  perturb {perturbed} -> detector {other}  {kind:5s} "
              f"max |delta| {gap:.6g}")
    print(f"\nworst cross-detector change {report['worst_cross']:.6g}")
    print(f"worst self change           {report['worst_self']:.6g}  "
          "(the probe reached the network)")

    if report["worst_self"] == 0.0:
        print("\nthe perturbation changed nothing at all, so this measured nothing")
        return 1
    if not report["separable"]:
        print("\nNOT SEPARABLE -- run with engine.use_frontend_cache=False")
        return 1
    print("\nSEPARABLE. The frontend cache is valid for this network on this device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
