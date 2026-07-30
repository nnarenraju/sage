# Closing the O3b→O3a Domain Gap: An Actionable SP Synthesis

**Problem restated for framing.** A per-detector multi-scale CNN is trained to detect a known injected transient in *real* period-B (O3b) noise and must generalize to *real* period-A (O3a) noise that differs in (a) broadband color, (b) narrowband lines (persistent + wandering), (c) non-stationarity + non-Gaussian glitches. We may **not** use A time-domain data but **may** freely estimate/use A PSDs/spectra. Everything must run per-sample on GPU (FFT-friendly, differentiable-adjacent a plus).

### Where things slot in the actual code (the map every technique below references)

```
real B noise (TD)
  └─ RecolourPostprocess.forward   sage/data/noise/recolour.py   [FD, @no_grad]
       • whiten by segment's own O3b ASD (per-run/segment bank)
       • × random real O3a ASD (bank ~250k/det, 16 GB/det resident)
       • dr_gain: k·σ(f)-bounded smooth linear+quad multiplicative jitter (within-manifold)
  └─ + injected signal (FD)
  └─ FiducialWhitening.forward     sage/dsp/whiten.py            [FD→TD, @no_grad, SEVERS autograd]
       • X_white = X_fd · 2Δf/(√0.5·ASD_fiducial);  ASD_fiducial = max(A,B) + GaussianSoftNotchBlackout
       • irfft → strip corrupted_len edges
  └─ [MinMaxNormalise]             sage/dsp/normalise.py         [in-graph, optional, per-sample per-det]
  └─ MultirateSampler (dyadic downsample)
  └─ MSCNN1D_2DResNetCBAM  norm=self.norm(x.float())  {group/instance/layer/batch}  networks.py
  └─ BCE (+ heteroscedastic point-estimate head)
```

Two facts constrain everything: **whitening is `@torch.no_grad()`** — any whitening/notch change is a *fixed* transform (fine for masks, no gradient concern); learnable front-ends (arcsinh-β, PCEN, MAD) must live at the **normalization slot**, which *is* in-graph. And the current "notch" is a *soft ASD-inflation* (`GaussianSoftNotchBlackout`) applied to the fiducial, **not** a residual-line removal on the whitened output — a distinction that turns out to be the crux of the line gap.

---

## Stage (i) — Whitening / adaptive whitening (ranked)

**1. FFT-domain per-bin line notch with local-median-floor replacement — HIGH.** *(Boll 1979 spectral subtraction; Thomson 1982 harmonic F-test as the detector.)* After dividing by the fiducial ASD, bins flagged as lines have their magnitude **replaced by the local-median broadband floor** instead of being divided by a (line-inflated) ASD. This is the frequency-domain dual of the entire notch-filter family and the cheapest, best-fit upgrade to what already exists.
- *Slot / compute / cost:* in `whiten.py`, immediately after `X_white = X_fd * whitening`, apply a fixed per-bin gain mask `g[f]` that maps flagged bins to `median(neighbours)` (soft cosine-tapered gain to avoid iFFT ringing). Line catalog = **union of A and B lines** built offline (Thomson F-test on B; peak-pick A PSDs). Pure elementwise multiply, fully batched, differentiable-adjacent, ~free.
- *Caveats:* it zeros any signal energy in the notched bins — keep notches as narrow as the measured linewidth. Use the **fixed union catalog** (identical mask train+inference) rather than a per-segment F-test, to preserve the identical-mask guarantee; add per-segment refinement only if wandering lines demonstrably survive. **This differs from and improves on `GaussianSoftNotchBlackout`,** which inflates the fiducial (division still leaves a scaled residual and over-amplifies signal sitting on the line).

**2. Per-bin running-median STFT temporal normalization ("spectral whitening") — HIGH.** *(Bensen et al. 2007, ambient-noise seismology; running-absolute-mean/median.)* Divide each STFT frequency bin's magnitude by a **running temporal median of that same bin**; persistent *and* slowly-wandering lines normalize to ~1, broadband color flattens, and A-vs-B line-amplitude differences cancel because each segment is divided by *its own* running level — the single strongest domain-invariance property in the whole survey set, and it uses **no A time-domain data**.
- *Slot / compute / cost:* `torch.stft` on the (whitened or pre-whiten) TD → per-bin sliding-window temporal median (`unfold`+`median`, or a soft-quantile surrogate for gradients) → divide → `istft` → existing min-max/multirate. GPU-cheap.
- *Caveats:* a **median** (not mean) is mandatory so a short transient/glitch is ignored; a long chirp that dwells in a bin for many seconds can be partially whitened away — **gate the window length ≫ expected signal support** and blend with the fiducial (λ-mix) to bound self-whitening. Validate recovered SNR of the *longest* injection. Data-dependent nonlinearity → verify it doesn't create a *new* B-vs-A asymmetry.

**3. Minimum-phase whitening / recolour via Kolmogorov spectral factorization — MEDIUM (mild speculative).** *(Kolmogorov 1939; Oppenheim & Schafer cepstral min-phase design.)* Build a causal min-phase filter whose |H|²=target/current via the real-cepstrum Hilbert relation, so whitening/recolour imposes color with a **causal phase** rather than the zero-phase magnitude divide.
- *Slot / compute / cost:* per (segment, target) compute `logR → irfft → causal lifter → rfft → exp`, all FFT, batched, differentiable. Drop-in for `whiten.py`'s kernel and/or the recolour multiply.
- *Caveats:* concrete motivation here — the pipeline `irfft`s **then crops `corrupted_len`**, so zero-phase whitening smears glitch/edge energy symmetrically across the crop boundary; min-phase **localizes** it. `R=A/B` must be floored at deep notches/lines or `logR` blows up. Whether the benefit is measurable given zero-phase "works" is the open question — cheap to A/B.

**4. Short-time segmented whitening with overlap-add (piecewise-stationary) — MEDIUM.** Replace the single ÷fiducial with per-frame ÷local-ASD, OLA'd — tracks intra-12 s non-stationarity the fixed fiducial ignores. *Caveat:* single-frame ASD is ~1 DOF/bin (high variance) and is estimated from signal+noise, so it self-attenuates the injection; only safe with heavy smoothing / noise-only companion estimate + fiducial fallback on loud frames. **Item 2 is the lower-variance, more line-selective realization of the same idea — prefer it.**

**5. Minimum-statistics / IMCRA per-segment adaptive whitening — MEDIUM→LOW.** *(Martin 2001; Cohen 2003.)* Track the local noise floor from temporal minima and whiten by it. GPU-feasible (`-maxpool1d(-logE)`), needs no signal-free data. But it **largely duplicates item 2 at higher tuning cost**, IMCRA's signal-presence gate is tuned-on-speech and fires on our *target* transient, and the sliding minimum can latch onto a long chirp. Keep as a fallback to item 2, always blended with the fiducial.

**Rejected for the forward path (all surveys agree):** full prediction-error whiteners — **NLMS/LMS, RLS/GAL lattice, time-varying AR/Burg, LSW-wavelet, adaptive-notch IIR cascades (Rao-Kung/Nehorai, Regalia), SOGI-FLL, Kalman line trackers, comb filters, wavelet shrinkage, spectral gating/subtraction.** Two fatal modes: (i) a whitener that removes *all predictable structure* eats the locally-quasi-sinusoidal chirp; (ii) sequential IIR/Kalman recursion is GPU-hostile. Their *only* legitimate use is **offline**: pre-cleaning augmentation noise or emitting a line catalog. The fixed-frequency ANF collapses exactly to item 1.

---

## Stage (ii) — PSD estimation (fiducial & per-segment) (ranked)

**1. Robust median/percentile Welch — HIGH.** *(Welch 1967; Percival & Walden 1993 percentile estimators.)* Per-bin **median** (with the log(2)≈0.693 bias correction) or a low percentile across Welch windows, instead of the mean, so a glitchy window can't inflate the ASD and chronically under-whiten.
- *Slot / compute / cost:* swap the mean-reduce for a median-reduce over the window axis when building the fiducial *and* the per-segment/recolour banks (`segment_psds`, `recolour_psds`). Offline, trivially batched. **Cheapest single win.**
- *Caveats:* persistent lines survive the median (present in all windows) — correct, but this is not a line tool; pair with stage-(i) item 1.

**2. Thomson multitaper (DPSS) + harmonic F-test line catalog — HIGH.** *(Thomson 1982; Percival & Walden 1993.)* K≈2NW−1 Slepian-tapered eigenspectra give a low-leakage PSD; the F-test regresses the complex eigen-coefficients to flag statistically significant sinusoids and estimate their (freq, amplitude). This is the **substrate that feeds items (i)-1, (iii)-1/3, (v)-2.**
- *Slot / compute / cost:* precompute `DPSS(N, NW=4, K=7)` once as a constant; K batched rFFTs via one einsum. Build A/B fiducial + recolour ASDs and the **persistent-vs-wandering line inventory** offline (zero signal-suppression risk — noise-only).
- *Caveats:* the F-test needs *complex* eigencoefficients; if "A spectra" is read as magnitude-PSD-only you degrade to peak-picking A (the B-side F-test is always available). Reshaping must leave a *smooth* background, not carve deep notches.

**3. Percentile-envelope fiducial (90th) instead of strict `max(A,B)` — HIGH/MEDIUM.** Replace the per-bin strict `max` union with a high-percentile envelope over robustly-estimated A and B segments, so a single glitchy A segment can't dominate the union and chronically under-whiten every sample. One-line change to fiducial construction (`blackout.py`/`make_fiducial.py`). Composes with item 1.

**4. Cepstral/homomorphic liftering for a line-free broadband envelope — MEDIUM.** *(Bogert-Healy-Tukey 1963; Oppenheim & Schafer.)* Low-quefrency lifter of the log-PSD yields a smooth broadband whitening envelope **without deep line-notches** (which ring after iFFT/crop and over-amplify signal near lines); the separated high-quefrency term is a compact line model. `DCT → mask → iDCT` along frequency, ~free. *Caveat:* liftering *discards* lines, so whitening with it leaves lines un-normalized — only safe if stage-(i)-1 or normalization handles them; non-harmonic/wandering lines leak into low quefrency.

**5. Adaptive time-bandwidth / multi-resolution + deliberately broadened line treatment — MEDIUM (design principle).** *(Walden; Thomson; Kay 1988.)* Use high-NW smooth resolution for the broadband floor and a finer pass around cataloged lines, combined per bin. The **durable insight**: because A lines *wander*, a sharp fiducial notch at A's measured frequency misaligns on a future period — so **deliberately broaden/shallow** the line treatment in the fiducial. Treat as a tuning layer on items 2–3, not a module.

**6. Spectral kurtosis / spectral flatness maps — MEDIUM (offline only).** *(Antoni 2006; Johnston 1988.)* SK spikes on any non-Gaussian transient; flatness separates tonal from broadband. **Safe roles only:** (i) offline, down-weight glitch-contaminated TF cells when estimating the noise-only banks; (ii) a cheap **A-vs-B SK/flatness diagnostic** to quantify the shift and steer augmentation. **Never** as an inline gate — it would flag the target.

**Rejected/low:** minimum-statistics as *fiducial* (orthogonal to A/B gap), Huberized/robust-Whittle (redundant once you median), AR/Burg parametric PSD (models sharp lines terribly), Capon/APES (per-band matrix inverse, offline niche superseded by the F-test). IMCRA dropped (speech-tuned gate).

---

## Stage (iii) — Noise augmentation & synthesis (ranked)

**1. Wandering-line augmentation via SMS (sinusoids-plus-noise) — HIGH.** *(Serra & Smith 1990, Spectral Modeling Synthesis.)* Synthesize each strong line as `aᵢ(t)·sin(2π∫fᵢ(t)dt)` with `fᵢ, aᵢ` drawn from small random walks over A's *measured* line bands/amplitudes, added onto recoloured-B noise. **This is the only technique in the entire set that can move a line's center in frequency over the 12 s** — no static-ASD multiply (recolour, morphing, randomization) can.
- *Slot / compute / cost:* offline peak-track A spectrograms → per-line (f0, drift range, amp range, width). In `recolour.py`, after the recolour multiply, add the synthesized lines. Batched sinusoid synthesis is cheap; cost is the offline catalog. **Augmentation-only → zero inference signal-suppression risk.**
- *Caveats:* **double-counting** is the trap — if you recolour with an A-ASD that already contains the static line *and* add a synthesized line, it's applied twice. Notch the catalog lines out of the recolour ASD (or inject only the wandering component). Start with the few dominant lines. Get the drift/amplitude statistics wrong → train on an unrealistic domain.

**2. PSD morphing / log-geodesic spectral interpolation — HIGH.** *(Pitié & Kokaram 2007 linear Monge-Kantorovich; Cuturi 2013 / Kolouri et al. 2017 for the OT variant.)* `ASD(f;t)=ASD_B(f)^{1−t}·ASD_A(f)^t` draws a continuum of intermediate colors; allowing **t>1 extrapolates past A toward future unseen periods**. Generalizes the discrete A-ASD bank + `dr_gain` into a dense family and controlled extrapolation.
- *Slot / compute / cost:* wrap recolour target selection; draw `t~U(0,t_max)`, one elementwise `pow/exp` per sample.
- *Caveats:* **apply the log-morph to the broadband (liftered) envelope only.** Applied to *line* bins it produces **ghost double-lines** (cross-fades amplitude at both the B-freq and A-freq simultaneously) — handle lines via SMS/notch. Unbounded `t>1` can go off-manifold.

**3. Domain-randomization extension (per-line + modest above-manifold) — HIGH.** *(Tobin et al. 2017.)* **Already partially deployed** as `dr_gain` (smooth k·σ(f)-bounded broadband jitter, *strictly within* the real-ASD manifold). The new, additive part: extend to **per-line amplitude/width/frequency-shift jitter** at cataloged line bins, plus a **bounded above-manifold** component so the net becomes invariant to line placement/color *beyond* the A-vs-B spread — which is exactly what future-period robustness requires.
- *Slot / compute / cost:* extend the existing `dr_gain` block in `recolour.py`; few cheap elementwise ops. Shares the line catalog with SMS.
- *Caveats:* above-manifold jitter must stay bounded or you generate unphysical spectra.

**4. GMM / clustered PSD-target sampling — HIGH.** *(Tai-Jia-Tang 2005 EM color transfer; Ferradans et al. 2014.)* Cluster A's log-PSDs (k≈8–16 states capturing quiet/glitchy/line-flare regimes); each draw samples a cluster then an ASD within it, optionally up-weighting rare loud-line states. The principled version of "multiply by a random A ASD" that guarantees coverage of A's **multimodal, non-stationary** spread rather than its average.
- *Slot / compute / cost:* offline clustering; per-sample it's a two-stage index draw — cheaper than the current uniform gather. Composes as the *target selector* for any recolour map.
- *Caveats:* need enough distinct A PSDs to populate rare clusters; **sample actual observed A ASDs** (or stay inside the empirical hull) — a Gaussian draw in log-PSD space can produce unphysical spectra.

**5. Time-varying / segment-wise recolour (non-stationarity bridge) — HIGH/MEDIUM.** *(Dahlhaus 1997 locally-stationary model.)* Impose a **smoothly-interpolated sequence** of A short-window ASDs across the 12 s (color drifts as in A), on top of B's real glitch structure. Directly trains (a)+(b)+(c) in *augmentation* (no inference signal risk). *Caveat:* realism/complexity risk — get the local-ASD drift statistics wrong and you train an unrealistic domain or trample B's non-Gaussian structure. Natural extension of items 2–4; pairs with stage-(i)-2 at inference.

**6. Timmer & Koenig random-phase Gaussian synthesis — MEDIUM (auxiliary only).** *(Timmer & Koenig 1995.)* One complex-Gaussian draw + iFFT → purely-Gaussian A-colored noise. Use as a **controllable-fraction auxiliary channel** mixed with real recoloured-B, to teach A's line/color signature *without* B-glitch confounders and interpolate between the "A-Gaussian" and "B-with-glitches" regimes. **Never a replacement for recolour** (destroys all glitch structure). Use the two-Gaussian (real+imag) draw to avoid the under-variance bug.

**Caveated / debunked:**
- **IAAFT — MEDIUM, and the "strictly-better-than-recolour" claim is FALSE here.** *(Schreiber & Schmitz 1996.)* IAAFT preserves only the amplitude *histogram* and its rank-remap **delocalizes glitches in time** — a *regression* on axis (c) versus current recolour, which keeps the real B segment and only reshapes color per-bin (no cross-frequency leakage, glitch morphology and time-localization intact). Pilot only if the heavy-tail *marginal* empirically matters more than glitch time-localization. AAFT, plain FT phase-randomization, and wavelet-IAAFT/PWIAAFT are dominated/dropped.
- **MKL/Bures-Wasserstein OT recolour + CSD-CORAL — MEDIUM→LOW.** *(Knott-Smith 1984; Pitié 2007; Sun et al. 2016.)* The theoretically-correct multichannel generalization of `sqrt(A/B)` recolour, but with **independent detectors the CSD is ~diagonal** so it collapses to exactly the current scalar recolour except at the handful of environmentally-correlated line bins. MKL (least sample displacement) is the right choice *if* you ever do a matrix recolour.
- **Quantile/histogram matching of per-bin coefficients — MEDIUM.** *(Hilger & Ney 2006.)* Captures heavy tails second-order CORAL/MKL provably cannot; apply to **noise-only** segments before injection (else it denoises the transient) and pool CDFs per-band. Doubles as a per-band input-norm layer (see stage iv).

---

## Stage (iv) — Input normalization (ranked)

**1. Median/MAD robust standardization (RobustScaler) — HIGH.** *(Huber 1964; Rousseeuw MAD, ×1.4826.)* Center on the median, scale by MAD — the direct fix to the min-max/std failure mode: over ~24k samples the merger is a tiny fraction, so a glitch spike can no longer collapse the whole waveform, and a single positive per-channel scalar preserves peak shape and inter-detector amplitude ratios (localization).
- *Slot / compute / cost:* replace/augment `MinMaxNormalise` at the whiten→multirate slot; two fused GPU ops, in-graph/differentiable.
- *Caveats:* **defends against glitches (c), NOT lines (b)** — a coherent sinusoidal line's MAD ≈ its std, so this does *not* cure the GroupNorm/InstanceNorm line-inflated-scale pathology (that's stage-(i)-1's job).

**2. Monotone companding — arcsinh / signed-log / μ-law — HIGH.** *(Lupton, Gunn & Szalay 1999 asinh magnitudes; ITU-T G.711 μ-law; Anscombe 1948.)* An odd, monotone, invertible squash: linear near zero (coherent few-σ signal band untouched) and logarithmic in the tails (glitch spikes compressed). `arcsinh` beats `log` (handles bipolar whitened strain + zero-crossings) and `tanh` (doesn't flatten the merger peak to a rail).
- *Slot / compute / cost:* single elementwise op **after** robust scaling: `x → arcsinh(x/β)`, `β` a per-detector *learnable* param init ~1 robust-σ. Fully differentiable, sign/phase-preserving.
- *Caveats:* needs the prior robust scale so β is in σ-units; too-small β compresses the peak-height the point-estimate head reads; does not remove lines.

**3. Winsorized / robust-quantile min-max (the *corrected* min-max) — MEDIUM.** *(Tukey 1977.)* Divide by robust quantiles (0.1%/99.9%) instead of true min/max, so a glitch saturates a rail rather than compressing the signal — literally the fixed form of the min-max currently under test. Pair with an arcsinh rail for differentiability. *Caveat:* `torch.quantile` is a full sort; tight quantiles can nick the peak; no help for lines.

**4. Score-function-shaped nonlinearity — MEDIUM.** *(Kassam 1988; Poor 1994; Miller & Thomas 1972.)* For a weak known signal in non-Gaussian noise, the locally-optimal front-end is `g(x)=−p′(x)/p(x)` (the noise-pdf score) — a redescending soft-limiter for heavy tails. **Value is to justify/tune the arcsinh curve's shape** by fitting the whitened-B amplitude pdf, rather than an ad-hoc choice. *Caveat:* the full sign/Wilcoxon rank variant discards absolute amplitude → kills peak-height + inter-detector ratios; use only the **amplitude-aware bounded limiter**.

**5. Huber / trimmed / winsorized M-estimation of location-scale — MEDIUM.** *(Huber 1964; Maronna et al. 2006.)* 1–2 IRLS steps (k=1.345, 95% Gaussian efficiency), a smoother alternative to pure MAD. Marginal gain once you median — reach for it only if MAD proves too jumpy across segments.

**6. CMVN + RASTA on an STFT-magnitude branch — MEDIUM.** *(Hermansky & Morgan 1994; Furui 1981.)* Per-bin log-mean subtraction nulls multiplicative broadband color (a) as an inference-time analogue of recolour; RASTA's IIR bandpass along time rejects slow line drift (b). *Caveat:* log-STFT-magnitude domain → needs a separate spectrogram branch (phase handling); per-bin variance norm can rescale the chirp track. Prototype as a domain-alignment side-branch.

**7. PCEN — MEDIUM, with a hard caveat (cross-survey conflict resolved).** *(Wang et al. 2017; Lostanlen et al. 2019.)* Trainable AGC+DRC that would track A-vs-B loudness drift and soften glitches with ~5 learnable params. **But on the primary bipolar strain path it is disqualified:** forming a positive envelope discards the zero-crossings/inter-detector polarity that carry coherence/localization cues, and the AGC EMA can flatten the chirp's amplitude growth. Only viable on an **STFT-magnitude feature branch**, not the strain path.

**Dropped:** per-sample ordered-quantile/rank Gaussianization (warps merger morphology, destroys inter-detector ratios — keep only a *shared* Yeo-Johnson λ, which just re-derives arcsinh); ZCA/Ledoit-Wolf & Tyler (independent detectors → near-diagonal covariance, misdirected); spectral gating/subtraction & myriad & wavelet shrinkage (denoise the sub-threshold signal).

---

## Stage (v) — Domain / covariance alignment (ranked)

**1. GMM clustered PSD-target sampling — HIGH** *(cross-listed from iii-4).* Best-aligned with the PSD-only constraint; the top distribution-level attack on non-stationarity (c).

**2. Cepstral log-spectral color/line separation with liftering — MEDIUM.** *(Bogert-Healy-Tukey 1963; Furui 1981.)* Split log-PSD into low-quefrency (broadband) and high-quefrency (lines); **match broadband A color exactly while separately randomizing/attenuating lines** — formalizes the `max(A,B)`+notch idea and trains the CNN to ignore line placement. All-FFT, ~free. *Caveat:* the split is imperfect (dense/wandering line forests leak into low quefrency).

**3. Quantile/histogram matching — MEDIUM** *(cross-listed from iii; Hilger & Ney 2006).* The genuine higher-order complement to second-order alignment — captures line/glitch heavy tails CORAL/MKL cannot represent.

**4. Minimum-phase PSD-matching filter — MEDIUM** *(cross-listed from i-3).*

**5. MKL/Bures-Wasserstein OT recolour — MEDIUM→LOW** *(cross-listed from iii).* Correlated-line bins only.

**6. Adaptive per-segment whitening (min-stat, blended with fiducial) — MEDIUM** *(cross-listed from i).* The most direct lever on (c) that pure covariance alignment can't touch — the CNN sees *local SNR*, which transfers to future periods.

**Dropped/low:** **CSD-CORAL, Riemannian affine-invariant recentering, subspace-alignment/Procrustes, Tyler M-estimator** — all degenerate to per-channel scaling for independent sensors; Riemannian recentering actively imposes an *average* A color (against the goal). **Wasserstein-Fourier OT line transport** and **VTLN/bilinear frequency warping** — teleport line mass / can't register multiple independently-moving lines and drag the signal band. The domain gap lives in the **per-channel frequency spectrum, not a 3×3 cross-detector covariance.**

---

## Part A — REDUNDANT with what we already do, or only marginal

- **Cross-channel covariance alignment (CORAL / MKL / Bures / Riemannian / Tyler / ZCA-Ledoit-Wolf).** Sensors are ~independent → CSD is near-diagonal → these collapse to the existing scalar `sqrt(A/B)` recolour except at a few correlated line bins. Misdirected: color/lines are per-channel frequency structure.
- **Domain randomization (Tobin).** Already live as `dr_gain`, and *by design within-manifold*. Only the **above-manifold + per-line** extension is new.
- **Frequency-domain periodogram bootstrap (Franke-Härdle / Kirch-Politis).** Subsumed by the resident **~250k real-A-ASD bank** — the empirical spread of real segments already exceeds a parametric bootstrap around one smoothed PSD.
- **Per-bin independent quantile draws for the recolour target — TRAP.** Breaks the cross-bin correlation of a real ASD; the current bank samples **whole real A-ASDs** (correlation preserved) + σ(f)-bounded jitter. Do **not** switch to independent per-bin draws.
- **AAFT / FT phase-randomization / circulant embedding / AR-LPC synthesis / wavelet-IAAFT.** Dominated by recolour (destroy glitch morphology, or FFT-recolour does the coloring more directly).
- **Robust-Whittle / Huber periodogram, Huber vs MAD.** Redundant once you median.
- **Per-sample rank/ORQ Gaussianization; spectral gating/subtraction; wavelet shrinkage; myriad; comb/SOGI-FLL/Kalman/RLS/LMS forward whiteners.** Either destroy the peak/inter-detector ratios, denoise the sub-threshold signal, or eat the chirp / are GPU-hostile.
- **`GaussianSoftNotchBlackout` (current soft ASD-inflation notch).** Partially redundant with — and strictly improved by — the FFT per-bin local-median notch (Part B), which removes the *residual* line rather than dividing by an inflated ASD.

## Part B — GENUINELY NEW, high-value (we are not doing)

1. **FFT per-bin line notch with local-median-floor replacement, union A∪B catalog from a Thomson F-test** — the direct line-gap (b) closer; different operation from our soft-inflate notch.
2. **Per-bin running-median STFT temporal normalization** — per-bin self-normalization → invariance to persistent+wandering lines, color, and non-stationarity, *including future unseen periods* (a)+(b)+(c).
3. **SMS wandering-line augmentation** — the only method that moves line centers; trains line-placement invariance.
4. **Robust MAD + learnable-β arcsinh input stack** — the corrected form of the min-max under test; fixes glitch-inflated scale (c).
5. **Recolour-as-a-distribution: log-geodesic morphing (+t>1) + GMM clustered targets + above-manifold line jitter** — future-period continuum (b).
6. **Robust median/percentile Welch + 90th-percentile-envelope fiducial** — near-free estimator/fiducial upgrade.
7. **Cepstral broadband/line separation** for a clean line-free fiducial background.
8. **Minimum-phase whitening/recolour** — causal phase, glitch localization vs symmetric ringing across the crop boundary (speculative benefit).
9. **Spectral-kurtosis offline glitch-masking of noise-only banks + A/B SK/flatness diagnostic** — safe, offline.
10. **Quantile/histogram-matching input-norm layer** — higher-order marginal alignment.

---

## TOP 5 EXPERIMENTS TO RUN NEXT (ordered)

> Prioritized for (a) closing the A line/non-stationarity gap, (b) future-unseen-period invariance, (c) composition with the running combined-fiducial + min-max/InstanceNorm arms. **Do the near-free fiducial upgrade — robust median Welch + 90th-percentile envelope — alongside Exp 1's offline F-test pass; it's a one-line bank-construction change.**

**Exp 1 — FFT-domain per-bin line notch (local-median floor), union A∪B catalog.** *(Boll 1979; Thomson 1982.)*
- **Hypothesis:** Replacing residual line bins with the local broadband-floor median (identical mask train+inference) removes the un-notched lines (e.g. ~1013 Hz) that inflate the InstanceNorm/GroupNorm/min-max scale, closing gap (b) and lifting the ~0.24→~0.185 BCE plateau.
- **Minimal implementation:** offline Thomson F-test (DPSS NW=4,K=7) on B noise + peak-pick A PSDs → per-detector union line-bin catalog. In `whiten.py`, after `X_white = X_fd*whitening`, apply a fixed cosine-tapered per-bin gain mask that floors flagged bins to `median(neighbours)`. Replaces/augments `GaussianSoftNotchBlackout`; runs in the existing `@no_grad` whitener; composes with every norm arm.
- **Confirming metric:** A-noise detection metric (sensitive distance at fixed FAR) ↑ and A-noise BCE ↓; per-bin whitened-noise variance at line freqs → ~1; InstanceNorm running-var no longer spikes at line bins (and the InstanceNorm-NaN failure disappears).

**Exp 2 — Per-bin running-median STFT temporal whitening, blended with the fiducial.** *(Bensen et al. 2007.)*
- **Hypothesis:** Dividing each STFT bin by its own running temporal median makes persistent+wandering lines and A-vs-B color/level cancel to ~1 using only each segment's own statistics → closes (a)+(b)+(c) and transfers to future periods with zero A time-domain.
- **Minimal implementation:** `stft → per-bin sliding-window temporal median (window ≫ longest BBH support) → divide → istft`, inserted before/after the fiducial with a λ-blend to bound self-whitening; then existing min-max/multirate. Soft-quantile surrogate if end-to-end gradients wanted (whitening is currently no_grad, so a fixed op is fine).
- **Confirming metric:** A-noise detection metric ↑ **while recovered matched-filter SNR of injections is preserved** (guard against chirp self-whitening); ideally test on a held-out third epoch as a future-period proxy.

**Exp 3 — Wandering-line augmentation (SMS) on the recolour path.** *(Serra & Smith 1990.)*
- **Hypothesis:** Injecting lines whose centers random-walk within A's measured bands (amplitudes from A's line distribution) onto recoloured-B noise trains the CNN to ignore line placement → robustness to A's wandering lines *and* to future lines at new frequencies.
- **Minimal implementation:** reuse Exp 1's offline catalog for per-line (f0, drift, amp, width). In `recolour.py`, after the recolour multiply, add `Σ aᵢ(t)·sin(2π∫fᵢ(t)dt)`; **notch the catalog lines out of the recolour A-ASD first** to avoid double-counting. Start with the few dominant lines. Augmentation-only → no inference signal risk.
- **Confirming metric:** A-noise detection metric, **conditioned on segments with active wandering lines**; ablate vs static-line recolour.

**Exp 4 — Robust input-norm: MAD standardization + learnable-β arcsinh, as the corrected min-max.** *(Huber 1964; Lupton et al. 1999.)*
- **Hypothesis:** Replacing per-sample min-max (a glitch sets the scale) with median/MAD centering + arcsinh tail-compression stops glitches collapsing the waveform and bounds A's unseen glitch amplitudes, improving (c) without harming peak-height/inter-detector ratios.
- **Minimal implementation:** new stage in `make_processor` parallel to `MinMaxNormalise` (in-graph, differentiable): `x←(x−median)/(1.4826·MAD+eps); x←arcsinh(x/β)`, per-detector learnable `β` init ~1σ. Head-to-head against the `config_HL_in_minmax` arm.
- **Confirming metric:** A-noise detection metric on glitchy segments ↑; per-sample input-scale variance (A vs B) shrinks; beats the min-max arm head-to-head.

**Exp 5 — Recolour target as a distribution: log-geodesic morphing (t>1) + GMM-clustered A sampling + above-manifold line jitter.** *(Pitié & Kokaram 2007; Tai-Jia-Tang 2005; Tobin et al. 2017.)*
- **Hypothesis:** Densely filling the B→A color continuum and deliberately extrapolating past A buys graceful degradation on future unseen periods, beyond snapping to the discrete real-A bank.
- **Minimal implementation:** in `recolour.py` target selection — (i) cluster log-A-ASDs (k≈8–16), sample cluster-then-ASD; (ii) draw `t~U(0,t_max>1)`, `ASD_target = ASD_B^{1−t}·ASD_A^t` on the **broadband (liftered) envelope only**; (iii) leave lines to Exp 3/notch (no log-morph on line bins → avoids ghost double-lines); (iv) extend `dr_gain` with a bounded above-manifold term.
- **Confirming metric:** A-noise metric **and** a held-out third-epoch (future-period proxy) metric ↑; verify morphed spectra stay physical (no ghost lines, no negative/blown-up bins).

**Speculative flags:** min-phase whitening (Exp-adjacent; benefit unproven), PCEN on raw strain (disqualified — envelope discards phase/polarity), Wasserstein-Fourier line transport (teleports spectral mass), time-varying recolour realism (Exp 5's drift statistics must be validated against A spectra). **Exp 1 + 4 are the cheapest and compose immediately with the current combined-fiducial + min-max/InstanceNorm runs; Exp 2/3/5 are the deeper (b)+(c) and future-period gap-closers.**