IMRPhenomPv2 on GPU
====================

:class:`~sage.data.waveform.approximants.IMRPhenomPv2.IMRPhenomPv2` extends
IMRPhenomD to precessing-spin binaries using the PhenomPv2 "twist-up" formalism.
It is the **production signal sampler** in Sage — wrapping parameter sampling,
waveform generation, multi-detector projection, and SNR rescaling into a single
callable that the training loop invokes once per batch.

Parameters
----------

Each waveform is characterised by 15 physical parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Index
     - Parameter
     - Description
   * - 0
     - ``m1``
     - Primary mass in solar masses
   * - 1
     - ``m2``
     - Secondary mass in solar masses
   * - 2–4
     - ``s1x, s1y, s1z``
     - Primary spin components (each −1 to +1)
   * - 5–7
     - ``s2x, s2y, s2z``
     - Secondary spin components
   * - 8
     - ``dist_mpc``
     - Luminosity distance in Mpc
   * - 9
     - ``tc``
     - Time of coalescence in seconds
   * - 10
     - ``phic``
     - Coalescence phase in radians
   * - 11
     - ``inclination``
     - Inclination angle in radians
   * - 12
     - ``polarization_angle``
     - Polarisation angle in radians
   * - 13
     - ``ra``
     - Right ascension in radians
   * - 14
     - ``dec``
     - Declination in radians

Setting up the signal sampler
------------------------------

Pass a parameter sampler, a projection method, and an SNR rescaler at construction
time. Sage handles the rest — no manual frequency-grid construction or explicit
parameter batches are needed:

.. code-block:: python

    from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2
    from sage.data.waveform import HalfNorm
    from sage.data.waveform.snr import OptimalSNRRescaler

    # Parameter prior from the YAML config
    param_sampler = read_from_config("./gwconfig.yaml", seed=150914)

    # ConstantProjection applies time-independent antenna-pattern weighting
    # for each detector — no manual RA/dec/polarisation passing needed
    waveform_project = ConstantProjection()

    # Half-normal SNR prior: most injections fall between SNR 5 and 20
    snrscaler = OptimalSNRRescaler(HalfNorm(scale=4.0, loc=5.0, seed=150914))

    signal_sampler = IMRPhenomPv2(param_sampler, waveform_project, augment=snrscaler)

Calling the sampler
--------------------

A single call samples parameters from the prior, generates precessing waveforms on
the GPU, projects them onto each detector, and applies the SNR rescaler:

.. code-block:: python

    signal_data, signal_targets = signal_sampler()
    # signal_data:    (S, n_detectors, n_freq) — complex FD strain per detector
    # signal_targets: (S, num_pe + 1)         — regression targets + class label (1)

where ``S = int(batch_size * class_balance)`` (default 50% of the batch).
The frequency grid is determined automatically from the data config.

See :doc:`../bbh_params/gwconfig` for the full prior specification and
:doc:`../full_run` for a complete production example.

Comparison with LALSuite
-------------------------

Sage's IMRPhenomPv2 is validated against ``lalsimulation`` with a mismatch below
**2 × 10⁻⁵** against an ``aLIGOZeroDetHighPower`` PSD.  To reproduce the comparison,
generate a waveform via ``get_hphc`` with known parameters and compare against LAL:

.. code-block:: python

    import lal, lalsimulation as lalsim
    import pycbc
    from pycbc.filter import optimized_match
    from pycbc.psd import aLIGOZeroDetHighPower

    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomPv2")
    hp_lal, _ = lalsim.SimInspiralChooseFDWaveform(
        30.0 * lal.MSUN_SI, 29.0 * lal.MSUN_SI,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        440e6 * lal.PC_SI, 0.2, 0.2,
        0.0, 0.0, 0.0,
        1.0 / 16.0, 20.0, 2048.0, 20.0,
        None, approximant,
    )

    deltaf = 1.0 / 16.0
    n_bins = len(hp[0])
    psd = aLIGOZeroDetHighPower(n_bins, deltaf, 20.0)

    hp_lal_fs = pycbc.types.FrequencySeries(hp_lal.data.data[:n_bins], deltaf)
    hp_sage_fs = pycbc.types.FrequencySeries(hp[0].detach().cpu().numpy(), deltaf)

    match, _ = optimized_match(hp_lal_fs, hp_sage_fs, psd=psd, low_frequency_cutoff=20.0)
    print(f"Mismatch: {1 - match:.2e}")
    # Mismatch: 2.00e-05

.. note::

   The small residual mismatch (~2 × 10⁻⁵) arises from float32 vs float64 arithmetic
   in the Pv2 "twist-up" rotation matrices. It is well within the accuracy required
   for detection-focused training.
