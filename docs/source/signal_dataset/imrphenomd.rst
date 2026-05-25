IMRPhenomD on GPU
=================

:class:`~sage.data.waveform.approximants.IMRPhenomD.IMRPhenomD` is a fully vectorised,
``torch.compile``-compatible implementation of the IMRPhenomD aligned-spin
frequency-domain waveform model. It computes batches of :math:`h_+` and :math:`h_\times`
polarisations from a batch of physical parameters.

.. note::

   In production, signal generation is handled through
   :class:`~sage.data.waveform.approximants.IMRPhenomPv2.IMRPhenomPv2`, which extends
   IMRPhenomD to precessing spins and wraps parameter sampling, waveform generation,
   multi-detector projection, and SNR rescaling into a single callable. See
   :doc:`imrphenompv2` for the recommended production usage.

Parameters
----------

Each waveform is characterised by 11 physical parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Index
     - Parameter
     - Description
   * - 0
     - ``m1_msun``
     - Primary mass in solar masses
   * - 1
     - ``m2_msun``
     - Secondary mass in solar masses
   * - 2
     - ``chi1``
     - Aligned spin of the primary (−1 to +1)
   * - 3
     - ``chi2``
     - Aligned spin of the secondary (−1 to +1)
   * - 4
     - ``dist_mpc``
     - Luminosity distance in Mpc
   * - 5
     - ``tc``
     - Time of coalescence in seconds (relative to segment start)
   * - 6
     - ``phic``
     - Coalescence phase in radians
   * - 7
     - ``inclination``
     - Inclination angle in radians
   * - 8
     - ``polarization_angle``
     - Polarisation angle in radians
   * - 9
     - ``ra``
     - Right ascension in radians
   * - 10
     - ``dec``
     - Declination in radians

Comparison with LALSuite
-------------------------

Sage's IMRPhenomD is validated against ``lalsimulation.SimInspiralChooseFDWaveform``.
The two agree to floating-point precision (residuals ≈ 0) for the plus polarisation:

.. code-block:: python

    import lal, lalsimulation as lalsim

    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")
    hp_lal, hc_lal = lalsim.SimInspiralChooseFDWaveform(
        30.0 * lal.MSUN_SI, 30.0 * lal.MSUN_SI,
        0.0, 0.0, 0.99, 0.0, 0.0, 0.99,
        440e6 * lal.PC_SI, 0.5, 0.5,
        0.0, 0.0, 0.0,
        1.0 / 20.0, 20.0, 2048.0, 20.0,
        None, approximant,
    )

    # Residual between Sage and LAL
    import matplotlib.pyplot as plt
    plt.plot(hc_lal.data.data[:40_000] - hc[0].detach().cpu().numpy()[:40_000])
    # Residual is at absolute zero (bit-identical within float64 precision)

.. note::

   LALSuite zero-pads its output to the next power of two. A requested length of
   40 961 samples (20 s at 2048 Hz + DC bin) will be returned as 65 537 samples.
   Sage returns exactly the requested length.
