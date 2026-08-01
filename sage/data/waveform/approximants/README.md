# Waveform approximants — provenance and licensing

## What this code is

The approximants in this directory are GPU-native, batched PyTorch
implementations written for Sage. They are validated directly against LALSuite
(`lalsimulation`), not against any intermediate implementation: see
`tests/test_waveform_all_approximants.py`, which asserts a per-approximant
mismatch bound against LALSim across randomised draws from the production
priors.

Relative to any prior reference implementation, this code:

- is a **full reimplementation in PyTorch**, not a mechanical translation. The
  third-party reference covering parts of IMRPhenomD and IMRPhenomPv2 is
  written in JAX; the control flow, memory layout, batching and branch-free
  constructions here were written from the physics for PyTorch;
- is designed for **batched GPU evaluation under `torch.compile`**, which
  constrains the implementation in ways the reference does not share;
- **adds functionality** with no counterpart in the reference implementation;
- carries substantial **performance and efficiency work**; and
- **fixes correctness bugs**, including bugs present in the third-party
  reference implementation. Corrections are documented at the point of change.

IMRPhenomXAS and IMRPhenomXAS_NRTidalv3 are independent ports from the LALSuite
C sources and share no code with the MIT-licensed reference.

## Provenance chain

The chain of origin matters for licensing, and it is worth stating plainly:

1. **LALSuite** (LIGO/Virgo/KAGRA) is the origin of the IMRPhenomD and
   IMRPhenomPv2 models and their reference implementations. LALSuite is
   distributed under the **GNU GPL (v2 or later)**.
2. The **MIT-licensed third-party reference** is a JAX implementation that was
   itself written from those LALSuite models.
3. **Sage's implementation** is a PyTorch reimplementation validated against
   LALSuite directly, with the additions, performance work and corrections
   described above.

So the underlying algorithms in this directory descend from GPL-licensed
LALSuite code, not from an independent MIT-licensed origin. Sage is distributed
under **GPL-3.0-or-later**, which is consistent with that ancestry: GPL is the
governing license for a work derived from GPL sources, and MIT-licensed
material can be incorporated into a GPL work provided its notice is retained.

## Third-party licensing

Portions of **IMRPhenomD** and **IMRPhenomPv2**, together with the QNM data
arrays in **`phenom_data.py`**, derive from the MIT-licensed reference named
above.

Those components retain their original copyright notices and license terms.
The original license text is in `THIRD_PARTY_LICENSE` and **must not be removed
or altered**; attribution at the point of use is retained in the source files.

The MIT notice applies to those derived portions only, and only in part.
Everything else in this directory is part of Sage and is distributed under
GPL-3.0-or-later — see the top-level `LICENSE`.
