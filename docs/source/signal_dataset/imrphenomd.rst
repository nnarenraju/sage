IMRPhenomD on GPU
=================

:class:`~sage.data.waveform.approximants.IMRPhenomD.IMRPhenomD` is a fully vectorised,
``torch.compile``-compatible implementation of the IMRPhenomD aligned-spin
frequency-domain waveform model. It accepts a batch of parameter vectors and returns
batches of :math:`h_+` and :math:`h_\times` polarisations.

Parameters
----------

Each row of the parameter batch is a 1-D tensor with the following fields:

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

Generating a batch
-------------------

.. code-block:: python

    import torch
    import numpy as np
    from sage.data.waveform import IMRPhenomD as imr

    params = torch.tensor(
        [30.0, 30.0, 0.99, 0.99, 440.0, 0.0, 0.5, 0.5, 0.0, 0.12, 0.2],
        device="cuda:0",
        dtype=torch.float64,
    )

    batch_size = 500
    params_batch = params.unsqueeze(0).expand(batch_size, -1).clone()

    # Build the frequency grid
    f_l, f_u, del_f = 20.0, 2048.0, 1.0 / 20.0
    n = int(np.round((f_u - f_l) / del_f)) + 1
    f = (f_l + del_f * torch.arange(n, device="cuda:0", dtype=torch.float64))
    f = f.unsqueeze(0).expand(batch_size, -1).clone()

    f_ref = torch.tensor(f_l, device="cuda:0", dtype=torch.float64)
    f_ref = f_ref.unsqueeze(0).expand(batch_size, -1).clone()

    # Instantiate and run
    phd = imr.IMRPhenomD(f, f_ref)
    hp, hc = phd(params_batch, reproduce_lal=True)
    # hp, hc: complex128 tensors of shape (500, n_freq)

Performance
-----------

Without ``torch.compile``, 500 waveforms (each 20 s at 2048 Hz) take roughly 10 s on
first call. With compilation, steady-state throughput reaches about **3 100 waveforms
per second**:

.. code-block:: python

    import time

    compiled_phd = torch.compile(phd)

    out = 0.0
    for _ in range(1000):
        t0 = time.perf_counter()
        compiled_phd(params_batch, reproduce_lal=True)
        out += time.perf_counter() - t0

    per_waveform = (out / 1000) / batch_size
    print(f"Per-waveform time: {per_waveform:.6f} s")
    # ~0.00032 s  →  ~3124 waveforms/second

Multi-detector projection
--------------------------

Use :class:`~sage.data.waveform.project.ProjectWave` to project the polarisations onto
one or more detectors:

.. code-block:: python

    from sage.data.waveform import project

    # Build a full frequency grid starting from 0 Hz for projection
    f_l_proj = 0.0
    n_proj = int(np.round((f_u - f_l_proj) / del_f)) + 1
    f_proj = f_l_proj + del_f * torch.arange(n_proj, device="cuda:0", dtype=torch.float64)

    pwave = project.ProjectWave(detector_names=("H1", "L1", "V1"), device="cuda:0")

    strain = pwave.constant_project(
        hp, hc, f_proj,
        ra=params_batch[:, 9],
        dec=params_batch[:, 10],
        polarization=params_batch[:, 8],
    )
    # strain: shape (500, 3, n_freq)

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
