# Search bug log

Every defect found in `sage/search/` or in the sgwc-1 code it is ported from, whether or
not it is fixed yet. **Nothing is closed by being forgotten.** A bug leaves this file only
by being fixed, with the fix named, or by being shown not to be a bug, with the evidence.

Fixing is a **separate step** from porting. A port is checked against sgwc-1's output, and
that check is impossible if the port silently corrects the original — so a bug inherited
from sgwc-1 is reproduced faithfully, recorded here, and fixed in its own change with a
test that fails before it and passes after.

Status: `OPEN` (found, not fixed) · `PINNED` (reproduced deliberately, fix pending) ·
`FIXED` (fixed, with the change named).

---

## OPEN / PINNED

### SB-19 — p_astro's KDE bandwidth is set by the range, not the data, so the noise density never converges · OPEN

**Where:** `sgwc-1/notebooks/pastro.ipynb` cells 13 and 22,
`build_noise_likelihood` / `build_signal_likelihood`.

**What:** both build the density as a KDE over uniformly spaced **histogram bin centres**,
weighted by the histogram:

```python
bin_edges = np.linspace(data.min(), data.max(), num_bins)
hist, edges = np.histogram(data, bins=bin_edges, density=True)
bin_centers = 0.5 * (edges[:-1] + edges[1:])
kde = gaussian_kde(bin_centers, weights=hist, bw_method='scott')
```

`gaussian_kde` derives its bandwidth from the spread of the points it is given. Those
points are uniform over `[min, max]`, so Scott's rule sees the variance of a *uniform
distribution over the range* rather than of the data.

**Verified:** on a peaked background with a long tail (σ = 1.83, range σ = 8.17) the
bandwidth comes out **4.95× too wide** and the peak density 22% low (0.287 against 0.368).

**Why it matters — this is the non-convergence.** The bandwidth tracks `max(data)`, and
the maximum of N Gaussian samples grows like √(2 ln N). So the density depends on *how
much background was collected*, not on the noise's shape. Measured on one fixed N(0,1)
population, varying only N:

| N background | max | h | p(10 \| noise) |
|---|---|---|---|
| 10,000 | 4.04 | 0.3381 | 2.7e-72 |
| 100,000 | 4.37 | 0.3439 | 5.3e-64 |
| 1,000,000 | 4.96 | 0.3584 | 1.0e-49 |
| 10,000,000 | 5.45 | 0.3615 | 3.2e-42 |

Thirty orders of magnitude at a fixed statistic, from the same population. A converging
estimator settles; this one does not. It is the mechanism behind p_astro failing to
converge as background is added, which is the behaviour that prompted this port.

**Fix:** take the bandwidth from the data. Not yet applied — porting first, per the rule.

### SB-20 — signal and noise densities treat their support differently · OPEN

**Where:** `pastro.ipynb` cell 22 against cell 13. The two builders are the same body
except that the signal one has its out-of-support line commented out:

```python
#pdf[(x < a) | (x > b)] = 0     # cell 22, signal
pdf[(x < a) | (x > b)] = 0      # cell 13, noise
```

**What:** beyond the support the signal density is still finite and the noise density is
exactly zero, so the likelihood ratio there is infinite by construction. **Any trigger
louder than the loudest background gets p_astro = 1 regardless of evidence.**

**Verified:** at 3 units past the support, `p_signal = 5.0e-13` and `p_noise = 0.0`.
Normalisation itself is fine (both integrate to 1.0000 within 1e-4).

Cell 41's `build_thresholded_likelihood` rebuilds both symmetrically, so which behaviour
reaches the final numbers depends on notebook cell order — which is itself the problem
worth removing in the port.

**Note on the intended fix:** `sage/search/tail.py` already fits a generalised Pareto tail
to the background and its docstring states that one tail model serves both the FAR
extrapolation and the p_astro noise density. Using it beyond the background's support is
the principled replacement for a hard zero, and needs discussing before it lands.

### SB-21 — the reparameterised prior is defined three times in one notebook, and the first is missing its Jacobian · OPEN (does not reach the published numbers)

**Where:** `pastro.ipynb` cells 36, 66 and 77, all defining
`mean_count_prior_alpha_lambda`.

| cell | returns | correct? |
|---|---|---|
| 36 | `1 / np.sqrt(lambda_s * lambda_n)` | **no** — missing the `lam` Jacobian |
| 66 | `lam / np.sqrt(lambda_s * lambda_n)` | yes |
| 77 | `lam / torch.sqrt(lambda_s * lambda_n)` | yes (but cell 77 contains a bare `raise` left from debugging and cannot run) |

**What:** changing variables from `(Ls, Ln)` to `(a, lam)` with `Ls = a·lam`,
`Ln = (1-a)·lam` carries `dLs dLn = lam · da dlam`. Cell 36 omits it, so its posterior
carries a spurious `1/lam` tilt. Cell 66 redefines the function correctly.

**Which one is live depends on cell execution order.** Cell 37 grids the posterior
immediately after cell 36, so that evaluation uses the wrong prior; cell 67 re-grids after
cell 66 using the right one, and that is the pass carried into the reported p_astro. So
the published numbers are unaffected — but the notebook contains a wrong intermediate that
looks like the real thing, which is the hazard the port removes by having one definition.

**Status in sage:** already correct. `pastro/rates.py::log_prior` applies the Jacobian
analytically and documents the cancellation.

### SB-22 — the monotonicity-policy vocabulary disagreed with the code that reads it · FIXED

**Where:** `sage/search/spec.py::MONOTONICITY_POLICIES` against
`sage/search/pastro/monotonic.py::apply_policy`.

**What:** the spec declared `("fail", "restrict", "remap")` and validated campaign configs
against it; `apply_policy` accepts `("stop", "restrict")` and raises on anything else. So
a campaign setting `monotonicity_policy = "fail"` — a value `spec.validate()` explicitly
blesses — passed validation and then died inside the p_astro stage, after the background
and the injections had been scored. `"remap"` was worse: it had been deliberately removed
from the implementation as having no counterpart in any source of truth, and the spec
still offered it.

**Found by:** writing the p_astro driver, which is the first code to call `apply_policy`
from a spec value.

**Fix:** the spec vocabulary is now `("stop", "restrict")`, pinned to the implementation
with the reasoning recorded beside it.

### SB-23 — the monotonicity verdict depends on the grid resolution, so `restrict` can fail to make a model monotone · DEFERRED

**Where:** `sage/search/pastro/monotonic.py::check_monotonicity` /
`largest_monotone_region`, and the `restrict` policy that uses them.

**Not a coding error.** `largest_monotone_region` is correct on the grid it is given:
measured on the 142 support nodes inside the region it returned, the log ratio is monotone
to machine precision.

**What is true:** the verdict is a statement about *those nodes*, not about the function.
On a 4× finer grid over the same interval the same densities are not monotone — worst
decrease 6.6e-4 in the log ratio. So:

- `check_monotonicity` can pass a model that is not monotone, if `support.n_stat`
  under-resolves it.
- `restrict` narrows the range while keeping `n_stat`, which *raises* the resolution. The
  densities are then rebuilt and re-checked at that finer spacing, and can fail again.
  Observed end to end in the p_astro driver: gate failed → restricted to
  [10.701, 13.824] → gate still failed, largest decrease 0.032.

**Why it matters:** the gate exists to decide whether FGMC is interpretable at all, and a
verdict that moves with a resolution parameter is not the decision it claims to be. It is
also self-defeating for `restrict`, whose whole purpose is to hand back a region the model
*is* interpretable on.

**Measured magnitude:** decreases of 6.6e-4 (wide) and 0.032 (after rebuild) in the log
ratio. Whether that is physically meaningful or numerical roughness in the KDE is the
question, and it decides the fix — a tolerance, a resolution-independent criterion, or
iterating restriction to a fixed point.

**DEFERRED (2026-08-20), and the reason widens the question.** The monotonicity gate is
not in sgwc-1: `grep -ril monoton` over `sgwc-1/notebooks/` returns nothing, and PyCBC's
three hits (`bin_utils`, `events/stat`, `waveform/ringdown`) are unrelated to a
density-ratio ordering check. So the whole gate is a Sage addition, and the first decision
is not how to fix its resolution dependence but **whether to keep it at all** — see
[[feedback-port-sgwc1-verbatim]]. It currently runs, records its verdict, and under
`restrict` narrows the support; nothing is blocked by it. Revisit once there are real
numbers to judge against.

### Deferred by ruling — figures without a reference

30 of the 47 declared figures have no counterpart in sgwc-1 or in PyCBC. Ruling of
2026-08-20: **keep what those two produce, record the rest as later**. Each is now marked
`deferred` on its own declaration with that reason, and each backed figure records where
it comes from in a new `FigureDecl.origin` field --
`"sgwc-1: pastro.ipynb, p_astro vs ranking statistic"`, `"pycbc: pycbc_page_ifar"` -- so
the claim is a pointer rather than an assertion.

To build (17): 10 from sgwc-1 (`statistic_ccdf`, `statistic_distributions`,
`pastro_densities`, `pastro_rate_posterior`, `pastro_rate_posterior_reparam`,
`pastro_curves`, `pastro_threshold_invariance`, `mass_plane_posteriors`,
`mchirp_q_coverage`, `event_spectrograms`) and 7 from PyCBC (`cumulative_vs_ifar`,
`far_versus_statistic`, `sensitive_distance`, `vt_versus_far`, `injection_recovery`,
`livetime_and_duty_cycle`, `range_over_time`).

Pinned by `tests/test_search_figspec.py::TestProvenance`, which fails on any figure that
is neither backed nor deferred -- so adding one means naming its origin or saying
explicitly that it waits. Two among the deferred are worth arguing for when this is
revisited: `trials_comparison` (the trials stage produces nothing else) and
`recovery_of_known_events` (the O3a validation gate).

### Ruled out — not a bug

`p_astro_single_event` (cell 84) and `p_astro_all_events` (cell 90) sum the integrand over
the rate grid with no `dr * dlambda` factor, and cell 90 computes `dr` and `dlambda` and
then never uses them, which reads as a dropped term. It is not: cell 59 normalises with
`posterior /= np.sum(posterior)`, making the posterior a discrete probability **mass**, so
the bare sum is the correct expectation. The unused variables are leftovers.

---

## FIXED

Defects found by the 2026-08-20 adversarial review of the M2 chain and fixed the same day.
Recorded because the same mistakes are easy to reintroduce.

| id | what | fix |
|----|------|-----|
| SB-2 | Stage fingerprints were hand-picked summary scalars, so a lattice shifted one sample, a reversed lag ladder, or a FAR curve divided by a different livetime all kept their fingerprint while the product changed | `sage/search/fingerprint.py`; every driver digests its product |
| SB-3 | `digest_h5` folded `created_utc`, so an unchanged re-run cascaded the whole campaign | `VOLATILE_ATTRS` excluded from the digest |
| SB-4 | `BackgroundSet.save`/`load` dropped `foreground_livetime_s`, so every removal mode's curve was divided by the un-vetoed exposure | persisted as an attribute; `far.run` uses the mode's own |
| SB-5 | `_declared_slides` counted a shard as covered because it was *stamped*, which happens at creation — a partial ladder collated | requires `finalised` and every committed block |
| SB-6 | `overdispersion_lrt` was fed ranking-statistic bins where sgwc-1 bins trigger *times*; reported a Poisson-by-construction background as over-dispersed at p = 0 | `far._time_binned_counts`, 10 s bins as `search.ipynb` cell 331 |
| SB-7 | The keep threshold was a read-modify-write over `slide_plan.h5`, which `save` truncates in place; nine of ten array tasks died on Lustre | moved to `background/keep_threshold.json`, created `O_EXCL` |
| SB-8 | Re-running `slides` wiped the frozen keep threshold without moving its fingerprint | threshold no longer lives in the plan |
| SB-9 | `separability()` perturbed by a constant offset, so any shift-invariant cross-detector coupling passed as separable | perturb by substitution, as a time slide does; `SharedScaleNorm` negative control |
| SB-10 | `run_search` had never executed: `read_from_config` missing its `seed` (×2), `as_config` unimported, sampler encoding buffers never compiled | `engine.build_param_sampler` |
| SB-11 | `as_config` handed the checkpoint's `dtype` through as the string `'torch.float32'` — the production O3b checkpoint stores exactly that | rehydrated to a `torch.dtype` |
| SB-12 | The search ran on the *training* run's device and fiducial spectra: `apply_shadow_overrides` was never called on the live path | called in `run_search` before the configs are registered |
| SB-13 | `keep_stream` was read via `getattr(writer, "keep_stream", False)` against a private attribute, so it was always False and the stream was never written | public property; `add_stream` given the statistic, not a 2-D array |
| SB-14 | `--n-slides 8` shared its tag, and so its directory, with the production campaign it smoke-tests, overwriting an 82-slide ladder in place | own tag and `out_dir`; `stages.claim_campaign_dir` |
| SB-15 | The spec hash depended on how `--config` was spelled, giving one campaign several identities | `SearchSpec.PROVENANCE_ONLY` |
| SB-16 | `runs/search/config_base.py` and `runs/o3b/config_base.py` shadowed each other in `sys.modules` under the bare name | `spec._unleak_siblings` |
| SB-17 | `submit.sh` drove every per-stage job with `run_search.py --stage`, which that driver rejects | `run_stage.py` |
| SB-27 | `CatalogueCache.put`/`_store` recorded the entry only in memory; the manifest was written by `freeze()`. A new process on the same cache root saw nothing and went to the network, so "fetched once, reproducible offline" was untrue after the first run | the entry table is persisted on every store; verified a fresh instance sees it |
| SB-28 | Nothing could stop a catalogue run reaching the network, and one did during testing | `CatalogueSpec.offline` (default **on**) and `CatalogueCache(offline_only=)`, which refuses an uncached URL rather than fetching it |
| SB-29 | The catalogue comparison used the whole cumulative GWOSC list, so an O3a search reported **109 O4 events as missed**. An event outside the analysed time was never searched for | every catalogue is restricted to the campaign's analysed lattice before anything is called recovered or missed, and the restricted span is written onto its conventions so `coverage_mask` uses the same bounds |
| SB-30 | `overlap_sets`/`union_times` merged any entries within the match tolerance, including two from the *same* source. The crossmatch tolerance (1.0 s) is wider than the clustering window (0.35 s), so two of our own candidates that clustering had ruled distinct were counted as one event | a group never takes two entries from one source; merging them would overrule a catalogue about its own list. Cross-source matching is unchanged |
| SB-26 | `ExternalCatalogue.filter_bbh` read its lower bound with `extra.get("mass2_lower_bound", event.mass2)`. GWOSC's parser writes that key with a `None` value whenever the source published no bound, and `dict.get` returns the stored `None` rather than the default -- so the fallback to the point estimate never ran and **every** event was kept. GW190425 (m2 = 1.4) then sat in the BBH list, and the O3a recovery gate would have counted a BNS the search never looked for as a missed event | explicit `None` check before the fallback; verified a BNS is excluded, an unmeasured event kept, and a bounded event judged on its bound |
| SB-24 | Candidate names are second-resolution (`SGW190401_150301`), which is unique for published events and not for a search's candidate list -- clustering only separates candidates by 0.35 s. The name is the identity every later join uses, so `trials._records_by_name` refused a table with 402 candidates in 193 s | `naming.disambiguate`: earliest in a colliding second keeps the bare name, the rest take `-1`, `-2` in time order |
| SB-25 | `candidates.from_triggers` demanded a p_astro row for every candidate, but p_astro is fitted on a bounded support that the monotonicity policy can narrow further, so it legitimately scores only part of the trigger set | join records nan outside the scored range and still refuses a missing row *inside* it, which is a genuine mismatch |
| SB-50 | Injections carried the Power-Law + Peak **source-frame** masses straight into `mass1`/`mass2`, which a waveform generator reads as **detector-frame**: no `(1+z)` anywhere in `sage/search/injection/`, while the same redshift *was* spent on the luminosity distance. The set was therefore internally inconsistent -- redshifted in distance, unredshifted in mass -- placing a binary at its correct cosmological distance while leaving it too light for that distance by a median factor of **2.06** (median z = 1.055). Median chirp mass 8.62 against a detector-frame 17.40. Inherited from sgwc-1, which does the same | FIXED under **PyCBC's convention**, established from the installed 2.10.0 source: unqualified `mass1`/`mass2` are what the generator receives, while `srcmass1`/`srcmass2`/`srcmchirp` are separate parameters filed under "derived parameters (these are not used for waveform generation)" (`waveform/parameters.py:167-174, 203, 216-231`); the relation is `msrc = mdet / (1+z)` (`mchirp_area.py:134`); and a PyCBC injection file stores detector-frame masses plus a luminosity distance and neither a redshift nor a source-frame mass (`population/scale_injections.py:13`), the redshift being recovered by inverting the distance. `build_injection_table` now writes `m_source * (1+z)`; `mchirp`, `q` and `chirp_distance` follow, and `distance` is untouched. Measured: the ratio matches (1+z) exactly, and the chirp-mass training-prior cut now **keeps more**, 87.2% -> 91.8%, because source-frame chirp masses fell below the trained lower edge. **A deliberate deviation from sgwc-1** |
| SB-48 | `SearchEngine.run` re-derived the block partition as `max(block.duration_s)` over the reader's blocks. `duration_s` is a block's **wall span**, gaps included; the partition is budgeted in **livetime**. On the O3a lattice the largest wall span is 254,401 s against a 32,768 s budget, so the engine walked **5 blocks where the reader held 30**. Nothing failed — a block carries the span slice both sides index through, so every window was still scored and the histogram summed to the full lattice — but the shard recorded 5 completed blocks against a stamped `n_blocks` of 30, resume granularity became a fifth of the run, and one coarse block's frontend cache residency came to **117 GB on an 80 GB card**. Found by reading the zerolag report, not by any failure | the reader records the `block_seconds` it was built with, and `_blocks_of` takes the reader's own blocks rather than recomputing a partition. The first zerolag shard was discarded: its block ids name the coarse partition, so a resume under the fine one would skip five blocks of different data and re-score the rest |
| SB-49 | `sage_submit` had no `--array` option, so the flag fell through to the command position: the array spec was submitted as the job body and every task exited 127. The `background-array` target had been written but never run | `--array` parsed and passed through to sbatch |
| SB-45 | `grid.iter_block_detector` marched every follower run from its first sample by a fixed stride. True for a GPS lag, false for a lattice roll: the rolled targets jump wherever they cross a reference span boundary, so a run kept marching past the jump and read a stretch of strain no window was ever assigned — silently, because every index stayed inside the segment | runs are split at any discontinuity in the target sequence as well as at segment exits; verified against `lattice[(i+k) mod N]` over all 12,532,817 O3a windows, worst error **0 samples** |
| SB-46 | The slides fingerprint covered lags, livetimes and window counts — none of which distinguish two rolled plans, since every rolled slide reports the same livetime and the same window count. A reassignment of shifts to slide_ids kept the fingerprint while every shard on disk, named by slide_id, described a different pairing. Found by a test written for the ladder's equivalent property | `window_shift` digested per detector alongside `offsets_s` |
| SB-47 | `EngineSpec.batch_size` was 8192 on the assumption that a large card wants a large batch. Measured: throughput is flat 1,024 to 8,192 and *peaks at 4,096*; 8,192 costs 2.3x the memory for 1% less, and 16,384 raises `integer out of range` from `torch.max_pool2d` (32-bit indexing) rather than running | default 4096, with the measurement recorded on the field. A window is 32,768 samples per detector, so the GPU saturates on arithmetic long before VRAM |
| SB-44 | Two representative-point methods were built from the Thyme notebook, measured, and **removed**: `joint_map_gaussian_kde` (Scott's bandwidth, unstandardised — Thyme's own notes record it over-smoothing into the wrong mode in 3-D, and it refuses at 14) and `marginal_map` (sgwc-1's — assembles a point out of per-parameter modes the joint posterior need not support, then finds the nearest sample by unstandardised distance; 2.16 nats below the maximum, and its answer moved from sample 10677 to 8222 once the sampler's bookkeeping was excluded). Both reproduced their reference answers before deletion — 10677 exactly | `METHODS` is `("max_likelihood", "joint_map_kde")`, and neither point method is the default: `InjectionSpec.population_mode` defaults to `"marginalise"`. A bad option left reachable is one that gets used |
| SB-42 | The representative population was chosen by marginal MAP, assembling a point coordinate-by-coordinate that the joint posterior need not support, when the release publishes `log_likelihood` and the densest sample is therefore known exactly. Measured: sgwc-1's sample 10677 sits **2.16 nats** below the maximum, so the population it injected is ~9x less likely than the one the data prefers, with β 2.60 against 0.63 and ξ_spin 0.44 against 0.97 | four selectable methods, default `max_likelihood` (which maximises likelihood + prior, and they coincide here because this release's `log_prior` is constant). `marginal_map` is kept and still reproduces 10677 |
| SB-43 | A joint-MAP density estimate over *all* posterior columns puts the sampler's bookkeeping — `selection`, `pdet_n_effective`, `surveyed_hypervolume`, `log_10_rate` — into the definition of the densest point. Measured: including them moved the joint MAP from sample 11175 to 5789, a likelihood **5.3 nats** worse | `population_columns` drops the auxiliary and constant columns before any density estimate or distance. A deviation from Thyme, which keeps them |
| SB-38 | `sources/gwtc3_powerlawpeak.nearest_sample` picks the hyperposterior sample by **unstandardised** Euclidean distance over all 22 columns, so `rate` (~15) and `surveyed_hypervolume` (~4225) dominate the fourteen mass and spin hyperparameters (~0.03 to ~98). sgwc-1 imports `StandardScaler` in the same notebook and never applies it | ported verbatim and asserted by test, because it selected sample 10677 and therefore the population behind sgwc-1's `p(x \| signal)`. Changing it changes the injection set |
| SB-39 | `test_mass_ratio_follows_its_own_primary` compared the light/heavy median gap at `atol=1e-3`, which is *inside* the Monte-Carlo noise band (0.0005 to 0.010), so the strict xfail flipped to a pass at random. It only ever ran once `gwpopulation` was installed | threshold moved to 0.05, between the noise and the 0.256 a correct sampler gives; seeded, because injection 0's primary mass alone moves the median between 0.73 and 0.93 |
| SB-40 | `make_spec` handled a dict override for `data` but passed every other one to `dataclasses.replace` whole, so `injection=dict(hyperposterior_path=...)` set `spec.injection` to a plain dict and failed at the first attribute access several stages later | any dict given for a dataclass field updates its named fields, with unknown names refused by name |
| SB-41 | `InjectionSpec.staged_path` documented that the drawn parameter set is written and reread, and nothing ever wrote it — the injections existed only inside the process that scored them, so the parameters behind `p(x \| signal)` could not be read back | staged under the campaign, reused only when draw count, seed, hyperposterior digest and sampler column names all match |
| SB-1 | Every injection's mass ratio was drawn from injection 0's conditional CDF — `get_p_q_vec` returns the `(N, n_q)` matrix of per-injection `p(q \| m1)` and the next line read one row of it. The numpy variant lost it differently, keeping only its loop's last iteration. Measured: `corr(m1, q)` = 0.004 where the truncation `q_min = mmin/m1` makes coupling mandatory; **8.2%** of draws had `m2` below the population's own `mmin`, down to 2.2 M☉ out of a BBH population; the whole set's mass-ratio distribution was fixed by one random number, injection 0's primary mass, moving the median q between 0.73 and 0.93 across runs | `interp1d_rows` does the inverse-CDF lookup per row, and each row's CDF is normalised by its own endpoint. After: `corr(m1, q)` = −0.386, **zero** draws below `mmin`, and the empirical median q tracks the analytic conditional to 0.005 at every primary mass from 6 to 40 M☉. **Reproducing sgwc-1's exact injection set is no longer possible, which is intended** |
| SB-18 | A background array task reporting `collated: False` was recorded as a complete stage | `stages._reports_complete` |
| SB-52 | `NoiseSlices` read `grid.spans_by_detector[detector]` for every detector, but a lattice stores the **reference detector's spans alone** — a follower's windows live on its own segments at its own local offsets and are derived, never stored. The injections stage died on `KeyError: 'L1'` at the first call. It could only ever have worked on a single-detector network | `AnalysisGrid.runs_for_detector` yields one detector's runs over the whole lattice, the follower path going through the same offset mapping the engine uses. Verified on the real O3a lattice: both detectors enumerate exactly 12,532,817 windows |
| SB-53 | `NoiseSlices` opened strain as a `np.memmap` of `data_{det}_{run}.bin` — a fork of the reader's layout choice, and the wrong branch of it. The search-grade release is **segmented HDF5**, one dataset per segment, precisely so it could keep the events the training release removed. Injections would have been added to noise read from a file that does not exist | `reader.open_stream` extracted to module level and called by both, so the layout is decided once, by the sidecar, where it already was |
| SB-51 | `SearchSpec.hash` fingerprints strain as `data_{det}_{run}.bin` only. The search-grade release is `.h5`, so the strain leg matched nothing and the release entered the hash through its sidecar alone — a rebuilt strain file under an unchanged sidecar would resume every stage against data it had never read | PENDING, patch held at `scratch/search_dev/SB-51_spec_hash.patch`. Widening the hash changes it, which invalidates the whole journal; landing it while a campaign is mid-chain would re-run its GPU stages, so it goes in once the shakedown completes |
| SB-54 | `InjectionSet.build` called `self.generator(n)`. `IMRPhenomPv2.forward`'s only positional argument is **`return_theta`** — it takes no count, and produces `generator.B` signals fixed by the frequency grid it was built on. So `n` was read as a truthy flag, the generator returned a third tensor, and the batch was silently whatever `B` happened to be | the campaign batches at `InjectionSet.batch_size`, which reads `generator.B`, and calls `forward()` with no argument. A partial final batch is run over the batch *ending* at the table's end and only its last rows kept, so the remainder enters `p(x \| signal)` once rather than being dropped |
| SB-55 | The approximant emits projected strain in the **frequency domain**; `build` added it to time-domain noise and handed the sum to `SearchEngine.forward`, which transforms again. Caught only by a shape guard — `(8, 2, 16385)` against `(16, 2, 32768)` — that fired for the batch mismatch first | the noise is transformed with the engine's own convention (`norm="forward"`, float32) and the two are added bin by bin, which is where training adds them and where the whitening buffer's normalisation is defined. `SearchEngine.forward_frequency` is the scoring path and `forward` reaches it through the transform, so injections are scored by the same code as a search window rather than a second copy |
| SB-56 | The intrinsic draw was **not reproducible**: `sample_1D_torch` and the mass-ratio lookup consume torch's *global* generator, which no seed reaches. Measured — two calls with identical arguments and the same seed kept 56 and 58 of 64 draws with different digests. `NoiseSlices` documents that the campaign is seeded so a resumed run scores against the same noise, and `_staged_table` reuses a staged draw, but a fresh campaign directory drew a different injection set every time, so the set behind `p(x \| signal)` could not be reproduced or released | a `torch.Generator` seeded from the campaign seed is threaded through every inverse-CDF lookup. Only the stream is pinned; the population each block is drawn from is untouched, so the distribution is unchanged. An unseeded call still follows the global generator, asserted, so the fix does not silently pin every caller to one set. **A deliberate deviation from sgwc-1**, which is unreproducible in the same way — same class as SB-1 |
| SB-57 | `pastro._injection_stats` and `figdata/build_significance` each read `injections/injection_triggers.h5`. The campaign writes one shard **per stream**, `injection_triggers_00.h5` — so both would have raised `FileNotFoundError` on a file no campaign has ever produced, and had they found one they would have fitted the signal density on a single stream | `campaign.scored_shards`/`scored_stats` name the shards once, next to the writer, and read every declared stream. A test asserts no other module in the package spells the name |
| SB-58 | Four `AnalysisGrid.build` call sites — both engine paths, the injection campaign and trials — left `reference_detector` to the default, which is whichever detector the segment dict happened to list first. The lattice is defined in the reference detector's frame, so two stages disagreeing about it describe different windows while reporting the same livetime and the same count. The same four requested the coverage decomposition, which costs **374 s and 11.8 GB on O3a** and which only the `grid` stage reads — on a 252-slide array that is ~26 GPU-h and 11.8 GB of host RAM per task, spent on a discarded object | every site passes both explicitly, checked by an AST test over the package rather than by a fixture that would only exercise one of them |
| SB-59 | `crossmatch` joined candidates to published catalogues on `gps` — the analysis **window start** — while a catalogue publishes a **merger** time. On the O3a smoke campaign the two differ by **13.05 s**, twenty times the 1.0 s match tolerance, so the recovery gate reported `n_known=0` and every recovered event as a new discovery. The search had in fact recovered them: against `tc_gps` the five loudest candidates are GW190408_181802, GW190421_213856, GW190413_052954, GW190412 and GW190413_134308, matching to between **0.009 and 0.094 s**. This is the gate the whole search is validated by, and it was silently inverted — a perfect recovery read as a perfect miss | `crossmatch.merger_times` prefers `tc_gps` and falls back to `gps` only for a campaign whose engine carried no decoder and therefore estimated no coalescence time. After: **5 of 7** confident O3a BBH recovered, the misses being GW190403_051519 and GW190426_190642, both marginal |
| SB-60 | `TriggerWriter.complete_block` commits through `atomic_h5(mode="a")`, which **copies the whole shard** before appending to it. The injection campaign committed one block per generator batch, so the snapshot cost grew with the shard while the work it protected did not — the campaign was quadratic in its own length. Measured: 4.4 M injections at a 2,048 batch is 2,129 commits of a shard growing to 235 MB, about **250 GB copied to protect 235 MB of work**, a ~1.4 h tax on a 2.2 h job. Invisible at the 4,358-injection smoke size, where it is 3 commits | scoring batch and commit block are separated: `COMMIT_ROWS = 131_072`, rounded down to a whole number of generator batches. At 4.4 M that is ~33 commits and 4 GB, and a killed job replays at most 131 k injections. Numbers are unchanged — same injections, same noise, same order |
| SB-61 | `signal_density` and `noise_density` measured the KDE bandwidth with `bandwidth_from_data(samples)` on the **whole** sample set, then handed those samples to `TruncatedKDE`, which discards everything outside the common support. A bandwidth carries `n ** (-1/(d+4))` and a robust scale, and both are properties of the sample they are measured on — so the rule counted samples the estimator never uses and measured a spread it never sees. Measured on the O3a campaign: 4,358,292 injections spanning -6 to 31 give **0.0642** where the 8,980 inside the support give **0.1244**, a kernel **1.94x too narrow**, and the deficit does not shrink as the campaign deepens because the discarded samples enter `n` as well. The resulting ripple in the signal density is what the monotonicity gate was failing on, so the policy narrowed the support to [15.36, 18.17] and **only 2 of 37 candidates** were inside it — the rest, including the loudest, got `p_astro = nan` | `bandwidth_on_support` measures the rule on the retained subset, shared with the truncation through `in_support` so the two are taken over the same set. After: worst decrease in `log(p_s/p_n)` **12.31 -> 0.0163**, support [9.22, 18.04] rather than [15.36, 18.17], and **35 of 37** candidates scored |
| SB-62 | `submit.sh` had no `trials` target. It is a declared stage in `CORE_STAGES` and `candidates` depends on it, but the case arm listing the per-stage targets skipped it, so `./submit.sh trials` printed the usage text and exited 0. In a hand-built chain that is worse than an error: the surrounding loop captured no job id, and the **next** stage was then submitted with an empty `--dependency`, so `candidates` started immediately and would have read the products of the run it was meant to follow. It had happened once before with the background chain | `trials` added to the stage arm and to the usage text, and `runs/search/chain_stages.sh` now builds dependent chains with the job id checked at every link — it refuses to submit the rest of a chain rather than submit it undependent |
| SB-63 | `run_stage.py --force` bypasses the manifest's completeness check but passes nothing to the driver, and the drivers resume from their **own products**. Forcing `injections` after the frame fix therefore re-entered a campaign whose shard already recorded every block complete, scored **nothing**, and finished in 25 s — then recorded that over the good manifest entry, so the campaign's record went from `n_scored=4358292` to `n_scored=0` while the stale shard stayed on disk. A force that silently does nothing is worse than one that fails | the shard is now discarded whenever the staged parameter table is redrawn, which is the actual invariant: a shard's rows are indexed by row number into that table, so a redraw makes every scored row describe a different binary. `_staged_table` reports whether it reused or redrew, and the table's provenance gained `mass_frame`, so the source-frame table from before SB-50 no longer matches and the redraw — and the shard discard — happen on their own |
| SB-64 | The **fitted GPD tail was used to calculate p_astro**. `pastro/run.py` passed `tail=curve.tail` into `noise_density`, so the noise density above the loudest background event was a model extrapolated **15 units of statistic** past anything measured (background reached 16.94, support 32.44). The signal density genuinely falls away above ~18.5 — the network's ranking statistic saturates, and the injections show it: 5,192 land in [18.0, 18.5) against 188 in [19.0, 20.0). Above 18 the extrapolated noise fell more slowly than the real signal density, so `log(p_s/p_n)` turned over and a candidate at 18.66 scored **below** one at 17.89. Measured: 221 decreasing nodes of 511 with the tail against 27 without, worst decrease −0.275 against −0.012, **every** violation above 18.03. sgwc-1 fits no tail anywhere — zero mentions in `search.ipynb` or `pastro.ipynb` — and zeroes its noise density outside the observed background range | the tail is not passed; `far.FIT_TAIL = False` so no curve carries one; `far_extrapolated_of` raises by name rather than returning the counted rate under a fitted name. After: p_astro monotone across all 37 candidates and saturating — 0.9999 at the loudest, against 0.834 before |
| SB-65 | `figdata/build_significance.pastro_threshold_invariance` **re-implements the whole p_astro construction** — support, both densities, `fit_rates`, `assign_pastro` — rather than reading what the `pastro` stage persisted. It is the only builder that does; every other one loads a product. Its call does not match the driver's: it still asks for `tail=getattr(curve, "tail", None)` and omits the `background_livetime_s`/`foreground_livetime_s` the driver passes. The two agree numerically **only** because no curve currently carries a tail — turn `FIT_TAIL` back on and the published figure would assert a threshold-stability computed with a GPD while the published numbers were computed without one. The figure genuinely has to refit at each threshold, so the duplication cannot simply be deleted | `pastro.run.fit_at_threshold` returns `(support, densities, posterior, kept)` and is the only place p_astro is constructed; the driver calls it once at the campaign's threshold and the figure calls it per rung. A test asserts via AST that neither builds its own densities |
| SB-66 | `candidates.contamination` sums `1 - p_astro` over the table, so a single `nan` p_astro silently made `expected_terrestrial` and `expected_astrophysical` `nan` — and it did: the campaign reported `expected_terrestrial: NaN` while `expected_terrestrial_confident` stayed finite, because the nan candidates happened to sit outside that tier. A headline contamination number degrading to `nan` rather than refusing is the kind of value that gets read as zero or skipped over | the sums run over the scored candidates and `n_unscored` (and `n_unscored_<tier>`) says how many were left out, so a quoted number is always a number and never silently covers less than it claims |
| SB-67 | The p_astro densities were **kernel estimates over the raw samples**, which nothing in the field does: PyCBC histograms both (`population.fgmc_functions.log_rho_bg`/`log_rho_fg`), GstLAL histograms and then smooths, sgwc-1 runs its kernel over the *histogram bin centres* weighted by the counts. A raw-sample KDE follows individual samples wherever they are sparse, which is the top of the range -- where a detection lives. Measured on the test background at a Silverman bandwidth of 0.155: with no samples at all between 13 and 15 the log density read **-34.6** there and recovered to -9.4 two units later off a single sample. The fitted GPD tail had been masking this, so removing the tail is what exposed it | `HistogramDensity` ports PyCBC's construction -- counts over bin width over total, one fictitious count for an empty bin, Poisson fractional error carried alongside. `TruncatedKDE`, `bandwidth_from_data` and `bandwidth_on_support` deleted. At the same point the histogram reads -7.17 flagged at 100 per cent error: a conservative floor, marked as unmeasured |
| SB-68 | `figdata.build_significance.far_versus_statistic` passed `curve.is_extrapolated` -- the **bound method**, not a call -- to `np.asarray`, producing an object array no reader could use. The builder would have raised the moment the `figures` stage ran; it never had | called with the statistics to test. Found while removing the tail, not by any test |
| SB-69 | `pastro.validate.convergence_with_background` compared two credible-interval widths with an **absolute** 1e-12 tolerance. Interval widths span orders of magnitude between a marginal candidate and a saturated one -- 1e-1 against 6e-8 on the same campaign -- so the fixed allowance asked a saturated probe to narrow to one part in 1e4 while letting a marginal one widen by half. It reported `narrowing: False` on a probe whose interval was float noise | tolerance made fractional (`_REL_TOL = 0.05`), and `saturated`/`informative` reported so a caller cannot read a convergence pass off a candidate pinned at zero or one |
| SB-70 | `dataprep`'s prefetch looked ahead a fixed **three segments**. That fills the fetch pipeline when a segment spans the median four GWOSC files and starves it when segments are short — O3a's L1 runs through stretches of 0.16 h segments needing **one file each**, so three segments queued three files against **eight** fetch workers, leaving five idle and the 24-file staging area 87% empty. Measured mid-build: **1.12 MB/s against a 21 MB/s ceiling**. Not a redundancy defect — files are still fetched exactly once, and the existence check and eviction guard are both correct — purely a starved pipeline | the window is measured in files and extends until it holds `max_files`, which is what `prefetch` already truncates to. It walks at most `max_files` segments however short they are, so the cost stays bounded |
