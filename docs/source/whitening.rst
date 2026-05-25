Whitening Noisy Signals
========================

Whitening divides the frequency-domain strain by the square root of the one-sided PSD,
producing a time series whose noise floor is approximately flat and unit-variance.
Sage uses a *fiducial* PSD — the median of many noise draws — so the same whitening
kernel applies to every batch without per-segment recalculation inside the training loop.

Assembling a noisy batch
-------------------------

First generate a signal batch and a noise batch following the steps in
:doc:`signal_dataset/index` and :doc:`noise_dataset/index`, then add them:

.. code-block:: python

    # signal_batch: (B, n_detectors, n_freq) — complex FD, from IMRPhenomPv2
    # noise_batch:  (B, n_detectors, n_freq) — complex FD, from MemmapNoiseSampler
    noisy_signal = signal_batch + noise_batch

Loading the fiducial PSDs
--------------------------

The fiducial PSDs are written by :class:`~sage.data.primer.EstimatePSD` (see
:doc:`noise_dataset/fiducial_psds`). Load them once before training:

.. code-block:: python

    import json
    import numpy as np
    import torch
    from pathlib import Path

    psds_all = []
    for det in cfg.detectors:                          # e.g. ['H1', 'L1']
        psd_dir = Path(cfg.export_dir) / "fiducial_psds"
        psds_all.append(
            np.fromfile(psd_dir / f"fiducial_{det}_psd.bin", dtype=np.float64)
        )

    fiducial_psds = torch.from_numpy(np.stack(psds_all)).to(device=cfg.device)
    # shape: (n_detectors, n_freq)

Whitening
----------

.. code-block:: python

    from sage.dsp.whiten import FiducialWhitening

    whitener = FiducialWhitening(
        fiducial_psds,
        seq_len=int(data_cfg.sample_rate * (data_cfg.sample_length + 2 * data_cfg.padding_length)),
        sample_rate=data_cfg.sample_rate,
        device=cfg.device,
        corrupted_length=data_cfg.corrupted_length,
    )

    whitened = whitener.whiten(noisy_signal)
    # whitened: real TD tensor of shape (B, n_detectors, seq_len_trimmed)

:meth:`~sage.dsp.whiten.FiducialWhitening.whiten` performs the following steps
internally:

1. Multiplies the input by ``1 / sqrt(fiducial_psd * df)`` in the frequency domain.
2. Applies an inverse FFT to return a time-domain output.
3. Trims ``corrupted_length`` samples from both ends to remove roll-on/roll-off
   artefacts introduced by the finite-length whitening filter.

After whitening, the noise floor should be close to unit variance:

.. code-block:: python

    data = whitened[0][0].detach().cpu().numpy()
    print(data.mean(), data.std())
    # ~0.0  ~0.36   (std < 1 because only a band of the spectrum carries power)

Whitening pure noise vs. pure signal
--------------------------------------

Understanding the result of whitening each component separately is useful for
interpreting the network's input:

.. code-block:: python

    whitened_noise  = whitener.whiten(noise_batch)
    whitened_signal = whitener.whiten(signal_batch)

**Pure noise**

After whitening, a noise segment is approximately a zero-mean, stationary Gaussian
process with roughly unit spectral density. All spectral lines and the low-frequency
wall are suppressed, leaving a flat-spectrum noise floor. The typical appearance is
broadband irregular fluctuations with no visible structure.

**Pure signal**

A whitened GW signal has the chirp concentrated at the end of the segment (near the
coalescence time ``tc``). The whitening operation rescales the frequency content of
the signal by the noise curve, so high-frequency content near merger is amplified
relative to the long inspiral. At moderate SNRs (∼10–20), the chirp is barely visible
above the noise floor in a single whitened time series; it becomes clear only after
matched-filtering or network inference.

Visualising both together:

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(whitened_noise[0][0].detach().cpu().numpy(), alpha=0.5, label="noise")
    ax.plot(whitened_signal[0][0].detach().cpu().numpy(), c="k", linewidth=2.0, label="signal")
    ax.legend()
    plt.show()
