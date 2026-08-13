# Reference documents for `sage/search`

Every method in `sage/search` cites an equation in one of the documents below. Citations
name the local file and the equation number, so an implementation can be checked against
its source without leaving the repository.

The PDFs themselves are not checked in (see `.gitignore`). Fetch them with:

```bash
python docs/references/fetch.py
```

The registry in [`sage/search/references.py`](../../sage/search/references.py) holds the
same list in code, together with the specific equations this subpackage uses.
`sage.search.references.verify_all()` checks that every document is present.

| Key | arXiv | Title |
|---|---|---|
| `fgmc` | [1302.5341](https://arxiv.org/abs/1302.5341) | Counting And Confusion: Bayesian Rate Estimation With Multiple Populations |
| `unified_pastro` | [2305.00071](https://arxiv.org/abs/2305.00071) | A Unified p_astro for Gravitational Waves: Consistently Combining Information from Multiple Search Pipelines |
| `pycbc_search` | [1508.02357](https://arxiv.org/abs/1508.02357) | The PyCBC search for gravitational waves from compact binary coalescence |
| `sensitivity_injections` | [2508.10638](https://arxiv.org/abs/2508.10638) | Compact Binary Coalescence Sensitivity Estimates with Injection Campaigns during the LVK Fourth Observing Run |
| `beyond_gwtc3` | [2401.08709](https://arxiv.org/abs/2401.08709) | Beyond GWTC-3: Analysing and verifying new gravitational-wave events from community catalogues |
| `gwtc2p1` | [2108.01045](https://arxiv.org/abs/2108.01045) | GWTC-2.1: Deep Extended Catalog of Compact Binary Coalescences ... First Half of the Third Observing Run |
| `gwtc3` | [2111.03606](https://arxiv.org/abs/2111.03606) | GWTC-3: Compact Binary Coalescences Observed by LIGO and Virgo during the Second Part of the Third Observing Run |
| `gwtc4_methods` | [2508.18081](https://arxiv.org/abs/2508.18081) | GWTC-4.0: Methods for Identifying and Characterizing Gravitational-wave Transients |
| `gwtc4_results` | [2508.18082](https://arxiv.org/abs/2508.18082) | GWTC-4.0: Updating the Gravitational-wave Transient Catalog ... Fourth LVK Observing Run |
| `gwtc5_methods` | [2605.27224](https://arxiv.org/abs/2605.27224) | GWTC-5.0: Methods for Identifying and Characterizing Gravitational-wave Transients |

Titles were read from the first page of each fetched file rather than transcribed.

## Equations used

**`fgmc` — arXiv:1302.5341**

- Eq. (12): likelihood of the data given per-event foreground/background flags.
- Eq. (14): flag prior; an event is foreground with probability `Rf / (Rf + Rb)`.
- Eq. (21): rate posterior with the flags marginalised out,

  ```
  p(Rf, Rb, th | d, N)  ∝  Π_i [ Rf·f̂(x_i,th) + Rb·b̂(x_i,th) ] · exp[-(Rf+Rb)] · p(th) / √(Rf·Rb)
  ```

  The `1/√(Rf Rb)` factor is the Jeffreys prior on the two rates.
- Eq. (35): foreground-dominated limit `Rf^(N-1/2) exp(-Rf)`, peaked at `Rf = N - 1/2`.
  Useful as a closed-form check on any implementation.

**`unified_pastro` — arXiv:2305.00071**

- Eq. (4): count parameters, `Λ_s = R_s T` and `Λ_n = R_n T`.
- Eq. (10): rate posterior in count form,
  `p(Λs,Λn|{x},N) ∝ exp[-(Λn+Λs)]·π(Λs,Λn)·Π_i {Λs·p(x_i|S) + Λn·p(x_i|∅)}`.
- Eq. (11): per-trigger probability, marginalised over the rate posterior,
  `p_astro(x) = ∫ dΛs dΛn [Λs·p(x|S) / (Λs·p(x|S) + Λn·p(x|∅))] · p(Λs,Λn|{x},N)`.
- Section V: adopts a preliminary cut of `FAR ≤ 2 day⁻¹` when applied to real triggers.

Where a module implements one of these, its docstring names the file and the equation.
