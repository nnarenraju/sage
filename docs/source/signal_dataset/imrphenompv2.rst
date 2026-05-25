IMRPhenomPv2 on GPU
====================

:class:`~sage.data.waveform.approximants.IMRPhenomPv2.IMRPhenomPv2` extends
IMRPhenomD to precessing-spin binaries using the PhenomPv2 "twist-up" formalism.
It supports full 3-D spin vectors and is the default waveform model used for Sage
training.

Parameters
----------

Each row of the parameter batch carries 15 fields:

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

Generating a batch
-------------------

.. code-block:: python

    import torch
    import numpy as np
    from sage.data.waveform import IMRPhenomPv2
    from sage.data.waveform import read_from_config, ConstantProjection

    params = torch.tensor(
        [30.0, 29.0,           # m1, m2
         0.5, 0.5, 0.5,        # s1x, s1y, s1z
         0.5, 0.5, 0.5,        # s2x, s2y, s2z
         440.0, 11.1, 0.2,     # dist_mpc, tc, phic
         0.2, 0.0, 0.0, 0.0],  # inclination, polarization, ra, dec
        device="cuda:0",
        dtype=torch.float32,
    )

    batch_size = 512
    params_batch = params.unsqueeze(0).expand(batch_size, -1).clone()

    # Build the frequency grid
    f_l, f_u, del_f = 20.0, 1024.0, 1.0 / 16.0
    n = int(np.round((f_u - f_l) / del_f)) + 1
    f = (f_l + del_f * torch.arange(n, device="cuda:0", dtype=torch.float64))
    f = f.unsqueeze(0).expand(batch_size, -1).clone()

    f_ref = torch.tensor(f_l, device="cuda:0", dtype=torch.float32)
    f_ref = f_ref.unsqueeze(0).expand(batch_size, -1).clone()

    # Instantiate using a YAML prior config and a constant-sky projection
    param_sampler = read_from_config("./gwconfig.yaml", seed=0)
    waveform_project = ConstantProjection()
    php = IMRPhenomPv2(param_sampler, waveform_project)

    hp, hc = php.get_hphc(params_batch, reproduce_lal=True)
    # hp, hc: complex64 tensors of shape (512, n_freq)

Multi-detector projection
--------------------------

:class:`~sage.data.waveform.project.ConstantProjection` applies a constant (time-independent)
antenna-pattern projection and returns a detector-strain batch:

.. code-block:: python

    signal_batch = waveform_project(
        hp, hc,
        ra=params_batch[:, 13],
        dec=params_batch[:, 14],
        polarization=params_batch[:, 12],
    )
    print(signal_batch.shape)   # torch.Size([512, 2, 16385])

Performance
-----------

A batch of 512 precessing waveforms (16 s at 1024 Hz) completes in approximately
**0.19 s** on a single GPU (A100/V100 class), giving roughly **2 700 waveforms per
second**:

.. code-block:: python

    import time

    t0 = time.time()
    hp, hc = php.get_hphc(params_batch, reproduce_lal=True)
    print(f"Batch of {batch_size} Pv2 waveforms: {time.time() - t0:.3f} s")
    # Batch of 512 Pv2 waveforms: 0.187 s

Comparison with LALSuite
-------------------------

Sage's IMRPhenomPv2 is validated against ``lalsimulation`` with a mismatch below
**2 × 10⁻⁵** against an ``aLIGOZeroDetHighPower`` PSD:

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
