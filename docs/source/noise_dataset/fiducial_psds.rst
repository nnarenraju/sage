Fiducial PSDs & Recolouring PSDs
=================================

Sage needs two flavours of PSD on disk before training can start:

* **Segment PSDs** — one PSD per downloaded noise segment, used for per-segment
  whitening before recolouring.
* **Raw (recolouring) PSDs** — a large collection of PSDs estimated from random draws
  of the noise data; used to resample the spectral shape of a noise batch during
  training. The median of these forms the **fiducial detector PSD** used by the
  whitener.

Both are produced by :class:`~sage.data.primer.EstimatePSD`.

Setup
-----

.. code-block:: python

    from sage.data.primer import EstimatePSD, NoBlackout
    from sage.data.noise import MemmapSingleNoiseSampler
    from sage.dsp.welch import TorchWelch
    from sage.data.psd import smoothing

    detector = "H1"

    torch_welch = TorchWelch(
        delta_t=1 / 2048,
        seg_len=int(2048.0 * 4),       # 4-second Welch segments
        seg_stride=int(2048.0 * 2),    # 50% overlap
        avg_method="median",
    )

    psd_smoothener = smoothing.LogSplineSmoothing(
        smooth_factor=None,
        noise_low_frequency_cutoff=data_cfg.noise_low_frequency_cutoff,
    )

    epsd = EstimatePSD(
        detector=detector,
        apply_inverse_spectrum_truncation=False,
        max_filter_len=int(round(2048.0 * 2)),
        low_frequency_cutoff=15.0,
        trunc_method="hann",
        psd_method=torch_welch,
        num_samples=250_000,
        store_psds_as_bin=True,
        blackout_policy=NoBlackout(),
        interpolate_psd=True,
        training_sample_length=int(2048.0 * 16),
        psd_smoothener=psd_smoothener,
    )

Estimating segment PSDs
------------------------

This step computes one PSD per mini-segment in the ``.bin`` file. These are used for
per-segment whitening before any recolouring is applied.

.. code-block:: python

    epsd.estimate_segment_psds(
        noise_segments_file=f"./data_release/data_{detector}_O3b.bin"
    )
    # Computing PSDs per segment: 100%|██| 20286/20286

Output files are written under ``data_cfg.data_dir / "segment_psds"/``:

.. code-block:: text

    data_H1_O3b_psds.bin           ← concatenated float32 PSD array
    data_H1_O3b_psds_segments.json ← per-segment metadata (length, GPS times, …)

Estimating raw (recolouring) PSDs
-----------------------------------

This step draws ``num_samples`` random noise windows from the data, estimates a PSD
for each, and saves the full collection. The median of this collection becomes the
fiducial detector PSD stored separately.

.. code-block:: python

    noise_sampler = MemmapSingleNoiseSampler(
        f"./data_release/data_{detector}_O3b.bin",
        return_tensor=True,
    )

    epsd.estimate_raw_psds(
        noise_sampler=noise_sampler,
        duration=int(round(2048.0 * 16)),
    )

.. note::

   Estimating 250 000 PSDs is I/O-bound and takes several hours. Run it once and
   cache the output; it does not need to be repeated unless the noise data changes.

Output files are written under ``data_cfg.data_dir / "recolour_psds"/``:

.. code-block:: text

    raw_H1_psds.bin    ← (num_psds × num_freq_bins) float32 array
    raw_H1_psds.json   ← metadata: num_psds, num_freq_bins, delta_f, …

Reading PSD files
-----------------

**Recolouring PSDs:**

.. code-block:: python

    import json
    import numpy as np
    import torch
    from pathlib import Path

    psd_dir = Path(data_cfg.data_dir) / "recolour_psds"
    meta = json.loads((psd_dir / "raw_H1_psds.json").read_text())

    psds = np.fromfile(
        psd_dir / "raw_H1_psds.bin", dtype=np.float32
    ).reshape(meta["num_psds"], meta["num_freq_bins"])

    recolour_psds = torch.from_numpy(psds)

**Segment PSDs:**

.. code-block:: python

    psd_dir = Path(data_cfg.data_dir) / "segment_psds"
    meta = json.loads((psd_dir / "data_H1_O3b_psds_segments.json").read_text())

    raw = np.fromfile(psd_dir / "data_H1_O3b_psds.bin", dtype=np.float32)

    psds, cursor = [], 0
    for m in meta:
        n = m["psd_len"]
        psds.append(raw[cursor : cursor + n])
        cursor += n

    segment_psds = np.stack(psds)   # (N_seg, F)

Spline smoothing
----------------

The raw PSDs contain spectral lines (power-line harmonics, violin modes, etc.).
:class:`~sage.data.psd.smoothing.LogSplineSmoothing` fits a log-cubic spline to
each PSD to suppress these without distorting the broadband shape.

.. code-block:: python

    freqs = np.arange(meta["num_freq_bins"]) * (1.0 / 16.0)

    smoothener = smoothing.LogSplineSmoothing(
        smooth_factor=0.005 * len(freqs),
        noise_low_frequency_cutoff=data_cfg.noise_low_frequency_cutoff,
    )

    for psd in recolour_psds:
        psd_np = psd.detach().cpu().numpy().copy()
        smoothed = smoothener.smooth(freqs, psd_np)

Pass ``smooth_factor=None`` to use an automatic estimate based on the number of
frequency bins.
