Sampling Parameters & Plotting Distributions
=============================================

:func:`~sage.data.waveform.sampler.read_from_config` parses a ``gwconfig.yaml`` file
and returns a :class:`~sage.data.waveform.sampler.DistributionSampler` ready to
draw batches on the GPU.

Building the sampler
---------------------

.. code-block:: python

    from sage.data.waveform import sampler

    gwsampler = sampler.read_from_config("./gwconfig.yaml", seed=150914)

The ``seed`` is passed to a :class:`torch.Generator` that drives all random draws.
Using the same seed reproduces the exact same parameter sequence, which is useful for
debugging and for separating training and validation streams by convention (e.g.
seed 150914 for training, 170817 for validation).

Inspecting the sampler
-----------------------

.. code-block:: python

    # All parameter names in the output tensor (alphabetically sorted)
    print(gwsampler.param_names)
    # ['chirp_distance', 'coa_phase', 'dec', 'distance', 'inclination',
    #  'injection_time', 'mass1', 'mass2', 'mchirp', 'polarization', 'q',
    #  'ra', 'spin1_a', 'spin1_azimuthal', 'spin1_polar', 'spin1x', 'spin1y',
    #  'spin1z', 'spin2_a', 'spin2_azimuthal', 'spin2_polar', 'spin2x',
    #  'spin2y', 'spin2z', 'tc']

    # Mapping from name to column index
    get_idx = gwsampler.param_index
    print(get_idx["mass1"])       # e.g. 6
    print(get_idx["distance"])    # e.g. 3

Drawing a large batch
----------------------

.. code-block:: python

    batch = gwsampler(100_000)    # shape: (100_000, num_params)

    # Extract individual parameters
    mass1   = batch[:, get_idx["mass1"]].detach().cpu().numpy()
    mchirp  = batch[:, get_idx["mchirp"]].detach().cpu().numpy()
    distance = batch[:, get_idx["distance"]].detach().cpu().numpy()

Plotting all parameter distributions
--------------------------------------

The following code produces a single figure with one subplot per parameter, arranged
in a grid:

.. code-block:: python

    import math
    import numpy as np
    import matplotlib.pyplot as plt

    param_names = gwsampler.param_names
    n_params    = len(param_names)
    n_cols      = 5
    n_rows      = math.ceil(n_params / n_cols)

    batch_np = batch.detach().cpu().numpy()

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for i, name in enumerate(param_names):
        ax = axes_flat[i]
        ax.hist(batch_np[:, i], bins=80, color="steelblue", edgecolor="none", density=True)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("value", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", linewidth=0.3, alpha=0.5)

    # Hide any unused subplot panels
    for j in range(n_params, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("BBH prior distributions (100 000 samples)", fontsize=13)
    plt.savefig("bbh_prior_distributions.pdf", dpi=150, bbox_inches="tight")
    plt.show()

Expected shapes of individual distributions:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Expected shape
   * - ``mass1``, ``mass2``
     - Roughly flat over [7, 50] M☉ with a pile-up near equal mass from the
       ``mass_order`` constraint
   * - ``mchirp``
     - Peaked near low values (both low-mass edge and equal-mass configurations
       produce similar chirp masses)
   * - ``q``
     - Peaked near 1 (equal mass), falling toward high mass ratios
   * - ``distance``
     - ``∝ r²`` shape (rising toward the maximum after the chirp-distance transform)
   * - ``chirp_distance``
     - ``∝ r²`` uniform-in-volume shape over [130, 350] Mpc
   * - ``inclination``
     - Peaked near π/2 (edge-on; ``sin θ`` prior)
   * - ``dec``
     - Roughly flat (``cos δ`` prior on the sphere)
   * - ``ra``, ``coa_phase``, ``polarization``
     - Flat (uniform angle)
   * - ``spin1_a``, ``spin2_a``
     - Flat over [0, 0.99]
   * - ``spin1x``, ``spin1y``, ``spin2x``, ``spin2y``
     - Symmetric about zero (isotropic solid-angle prior)
   * - ``spin1z``, ``spin2z``
     - Flat over [−0.99, 0.99] (dominated by the uniform spin magnitude prior)
   * - ``tc``
     - Flat over [11.0, 11.2] s
   * - ``injection_time``
     - Flat over the O3a/O3b GPS range

Normalisation and standardisation
-----------------------------------

The sampler exposes helpers for normalising the output to ``[0, 1]`` (min-max
normalisation) or standardising to zero mean and unit variance. Both require
pre-computation of the statistics buffers first:

.. code-block:: python

    # Min-max normalisation (uses theoretical bounds)
    gwsampler._compile_batch_normaliser()
    normed = gwsampler.norm_from_batch(batch)         # (B, n_selected)
    back   = gwsampler.unnorm_from_batch(normed)       # recover physical values

    # Standardisation (uses empirical mean/std from 1M samples)
    gwsampler._compile_batch_standardiser(N=1_000_000)
    standardised = gwsampler.standardise_from_batch(batch)
    back         = gwsampler.unstandardise_from_batch(standardised)

These are used by the training loop to bring the regression targets into a
well-conditioned range before computing the heteroscedastic loss.
