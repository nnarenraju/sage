# Open items — `sage/search/`

Things known to be wrong, undecided, or unbuilt. Fixed defects live in `BUGS.md`; this is
what is still owed. Ordered by what blocks a production campaign.

Last measured 2026-08-25 against the O3a smoke campaign
(`/work/nagarajan/sage_runs/search/o3a_HL_smoke`, 14.52 d analysed, 0.358 yr background,
4,572,338 injections).

---

## Blocks a full run

### L-1 · `store` fails on its first execution
`UnknownColumn: unknown column of livetime 'foreground_s'`. The stage adapter writes a
column the `livetime` table does not declare; the store suggests `background_s`,
`zerolag_s`, `coincident_s`. Never run before, so never seen. **13 of 16 core drivers
work; this is one of the three that do not.**

### L-2 · `figdata` fails on its first execution
`figure 'cumulative_vs_ifar' is missing ['background_livetime_s', 'foreground_livetime_s']`.
The builder under-delivers against its own `FigureDecl.requires`. This is the
backward-requirements contract working as designed -- it is catching a real gap -- but the
gap has to be closed. Other builders may have the same shape; only this one has been
reached.

### L-3 · `sensitivity`, `figures`, `tables` have no `run()`
Three of sixteen core stages have no driver at all. `sensitivity` is deliberate -- VT was
deferred by ruling -- but `figures` and `tables` are the release path and nothing has been
written for them.

---

## Decisions owed

### L-4 · What `p(x | noise)` is above the loudest background event
Three conventions, none satisfying on a shallow background:

| | rule | loudest four candidates |
|---|---|---|
| sgwc-1 | floor at 1e-10 (`interp1d(fill_value=1e-10)`) | all exactly **1.0** |
| PyCBC | one fictitious count in a widened bin | 0.815, 0.943, 0.954, 0.939 -- **out of order** |
| removed | fitted GPD tail | worse inversion; see SB-64 |

Currently sgwc-1's floor. The user's objection stands: four candidates at exactly 1.0 is
degenerate.

**This resolves itself with background depth and is not really a convention question.**
The background tail e-folds every **1.99** in statistic (`d ln N/d stat = -0.502`, fitted
over stat 8.8-16.9), so the loudest background event grows as `log N`:

| livetime | events | loudest background |
|---|---:|---:|
| 0.358 yr (smoke) | 8,108 | 16.94 |
| **0.85 yr** | ~19,300 | **18.66** = our loudest candidate |
| 10 yr (production) | 226,622 | **23.58** |

At production depth every candidate sits inside measured background and none of the three
rules fires. Revisit after the first deep background rather than choosing now.

A fourth option worth considering meanwhile: mark such candidates *above the measured
background* rather than assign a number, as the FAR curve already does with
`is_extrapolated`.

### L-5 · Histogram bin width
`density.BIN_WIDTH = 0.5`, a module constant rather than a spec field (see L-7). Chosen
from occupancy: only **260** background events lie above the analysis threshold, giving 15
bins of ~17. PyCBC's own range is 0.1-0.5 against much deeper backgrounds.

Open: fix the width, or derive it from occupancy so it tightens as the background deepens.
At 10 yr there would be ~7,262 events above threshold and 0.1 would give ~96 per bin --
better resolution *and* less noise than 0.5 gives today. An occupancy rule adapts; a fixed
width does not, but a fixed width does not move the density when background is added.

**Not to be chosen by minimising out-of-order pairs.** That column is a symptom of having
260 events, and tuning against it fits the estimator to the answer.

### L-6 · Low-statistic p_astro ordering
Seven of 37 candidates are out of order at stat 9.3-11.6, p_astro 0.02-0.13 -- bin-to-bin
Poisson noise from those 260 background events. Expected to resolve at production depth
along with L-4/L-5. Recorded so it is not rediscovered as a defect.

### L-7 · The spec hash is global
Any change to `SearchSpec` invalidates all 16 stages, GPU included. This forced three
awkward choices in one session: the monotonicity policy left at a value nothing reads, the
histogram bin width put in a module constant rather than the spec, and a 5 GPU-hour re-run
to raise `n_draw`.

Per-stage hashing is the fix -- each stage keyed on the spec fields it actually reads, plus
its upstream fingerprints. **Not attempted:** under-declaring a stage's inputs silently
reuses stale products, which is far worse than over-invalidating. Needs a declaration per
stage and a test that a field a driver reads is one the hash covers.

---

## Deliberately parked

### L-8 · Monotonicity gate
`check_monotonicity` still runs and is reported; `apply_policy` is kept and tested but the
driver never calls it. Removed by ruling: a strict node-wise test on a finite-sample ratio
fails on estimator noise, and restricting on that discards the top of the support, which is
where the detections are.

### L-9 · SB-50, source vs detector frame
Settled -- detector frame, following PyCBC -- but it makes sage's injection set
deliberately different from sgwc-1's, which has the same source-frame bug. Recorded because
a future comparison against sgwc-1 numbers will not match and this is why.
