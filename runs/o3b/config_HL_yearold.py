#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""O3b HL -- YEAR-OLD-STYLE ablation (isolate the slow-loss cause).

Identical to the current production config_HL EXCEPT for the knobs we changed vs a
year ago, to test whether the slower loss decrease comes from optimiser/schedule/
notch rather than the architecture/data. Everything else (128 epochs, batch,
mining, recolour, EMA, InstanceNorm, ...) matches current production.

Changed back to "a year ago":
  * fiducials  : year-old MEDIAN PSDs (scratch/psds/median), regridded to the
                 current 16 s analysis grid, year-old blackout notches preserved
                 (fiducial_psds_yearold_median).
  * optimiser  : plain Adam (coupled L2), NOT AdamW.
  * weight decay: 1e-6 (vs 1e-4 now).
  * warmup     : none.
  * LR schedule: cosine annealing with warm restarts every 5 epochs (SGDR).
  * norm       : InstanceNorm (same as current config_HL).

Fresh + independent: its own export_dir and its own per-export_dir hard-mining
bank; touches neither the running A/B (prod_HL, prod_HL_gn) nor the old runs.

    ./submit.sh train_hard config_HL_yearold

__license__ = GPL-3.0-or-later
"""

from config_base import make_configs
from sage.core.config import register_configs
from sage.core.base_classes import BaseConfig, BaseDataConfig


def set_configs():
    cfg_cls, data_cfg_cls = make_configs(
        ["H1", "L1"],
        "/work/nagarajan/sage_runs/o3b/exp_HL_yearold",
        norm_type="instancenorm",
    )
    # --- year-old ablation overrides (everything else stays production) ---
    cfg_cls.fiducial_dir       = "/work/nagarajan/sage_runs/fiducial_psds_yearold_median"
    cfg_cls.optimizer          = "adam"      # plain Adam (coupled L2), not AdamW
    cfg_cls.weight_decay       = 1e-6        # vs 1e-4 in production
    cfg_cls.warmup_steps       = 0           # no warmup
    cfg_cls.warm_restart_t0    = 5           # cosine annealing + warm restarts every 5 epochs
    cfg_cls.warm_restart_tmult = 1           # fixed 5-epoch period (no lengthening)

    register_configs(BaseConfig(cfg_cls()), BaseDataConfig(data_cfg_cls()))
    print("Registered YEAR-OLD ablation cfg (Adam wd1e-6, no warmup, "
          "cosine warm-restart T0=5, year-old median fiducials, InstanceNorm)")
