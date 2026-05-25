Optimal Network SNR Estimation
================================

The optimal matched-filter SNR is the theoretical maximum SNR achievable for a known
signal in stationary Gaussian noise. Sage uses it to rescale injected waveforms so that
the training set covers a desired SNR distribution.

:class:`~sage.data.waveform.snr.OptimalSNREstimator` computes both the per-detector
optimal SNR and the combined network SNR from a frequency-domain signal batch.

.. code-block:: python

    from sage.data.waveform.snr import OptimalSNREstimator

    snr = OptimalSNREstimator()

    # signal_batch: (B, n_detectors, n_freq) — complex FD output from IMRPhenomPv2
    rho_net, rho_det = snr(signal_batch)

Output shapes
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Variable
     - Shape
     - Description
   * - ``rho_net``
     - ``(B,)``
     - Network (combined) optimal SNR: :math:`\rho_\text{net} = \sqrt{\sum_d \rho_d^2}`
   * - ``rho_det``
     - ``(B, n_detectors)``
     - Per-detector optimal SNR

.. code-block:: python

    print(rho_net.shape)    # torch.Size([512])
    print(rho_det.shape)    # torch.Size([512, 2])

    # Per-detector SNR for the first detector across the batch
    print(rho_det[:, 0])

For a batch of 512 injections at 440 Mpc with :math:`m_1 = m_2 = 30\,M_\odot`, the
individual detector SNRs typically range from ~6 (near-unfavorable sky location) to
~42 (optimal orientation).

SNR-based injection rescaling
------------------------------

In practice you do not call :class:`~sage.data.waveform.snr.OptimalSNREstimator`
directly during training. Instead, wrap it in
:class:`~sage.data.waveform.distributions.snr_rescaling.OptimalSNRRescaler`, which
draws a target SNR from a prior distribution and rescales each waveform amplitude
accordingly before injection:

.. code-block:: python

    from sage.data.waveform.distributions.snr_rescaling import OptimalSNRRescaler
    from sage.data.waveform import HalfNorm

    # Half-normal target SNR prior: mean ~5, most mass between 5 and 15
    target_snr_sampler = HalfNorm(scale=4.0, loc=5.0, seed=150914)
    snrscaler = OptimalSNRRescaler(target_snr_sampler)

Pass ``snrscaler`` as the ``augment`` argument when constructing the waveform generator
(see :doc:`signal_dataset/imrphenompv2`). The rescaler computes the current optimal SNR
of each waveform in the batch, then multiplies the amplitude by
``target_snr / current_snr``.
