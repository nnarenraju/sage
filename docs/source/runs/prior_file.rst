The Prior File
==============

``prior.yaml`` says how the parameters of the training signals are drawn.
Every training signal gets new parameters, drawn from this prior.

A complete prior file
---------------------

This is the ``bbh_narrow`` prior:

.. literalinclude:: ../../../sage/presets/priors/bbh_narrow.yaml
   :language: yaml

Parts of the file
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75
   :class: wrap

   * - Part
     - What it holds
   * - ``variable_params``
     - Every parameter the priors draw, listed once.
   * - ``priors``
     - One distribution for each parameter, or for a pair of angles.
   * - ``constraints``
     - Conditions every draw must meet; draws that fail are drawn again.
   * - ``waveform_transforms``
     - Parameters computed from the drawn ones, for the waveform.

``variable_params`` and ``priors`` are required; the other two are optional.
``variable_params`` must list exactly the parameters the priors draw, so
adding a prior means adding its parameters to the list.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 40 60
   :class: wrap

   * - Parameter
     - What it is
   * - ``mass1``, ``mass2``
     - The component masses, in solar masses.
   * - ``spin1_a``, ``spin2_a``
     - The size of each component's spin, from 0 to 1.
   * - ``spin1_polar``, ``spin1_azimuthal``, ``spin2_polar``,
       ``spin2_azimuthal``
     - The direction of each spin.
   * - ``ra``, ``dec``
     - The sky position: right ascension and declination.
   * - ``inclination``
     - The angle between the binary's orbit and the line of sight.
   * - ``coa_phase``
     - The orbital phase at merger.
   * - ``polarization``
     - The polarisation angle.
   * - ``chirp_distance``
     - The distance, scaled by the chirp mass, in Mpc.
   * - ``delta_tc``
     - When the merger happens, in seconds from the start of the window.

Distributions
-------------

Each prior names its distribution with ``name``:

.. list-table::
   :header-rows: 1
   :widths: 32 30 38
   :class: wrap

   * - ``name``
     - Settings
     - Draws
   * - ``uniform``
     - ``min``, ``max``
     - Uniformly between ``min`` and ``max``.
   * - ``uniform_angle``
     - none
     - An angle uniform from 0 to 2π.
   * - ``sin_angle``
     - none
     - An angle from 0 to π, with density proportional to its sine.
   * - ``uniform_sky``
     - ``polar-angle``, ``azimuthal-angle``
     - A sky position uniform over the sphere. The angles are named
       ``dec`` and ``ra`` unless set.
   * - ``uniform_solidangle``
     - ``polar-angle``, ``azimuthal-angle``
     - A direction uniform over the sphere, such as a spin's.
   * - ``uniform_radius``
     - ``min``, ``max``
     - A distance uniform in volume: density proportional to its square.
   * - ``uniform_comoving_volume``
     - ``min``, ``max``, ``cosmology``
     - A distance uniform in comoving volume. ``cosmology`` defaults to
       ``Planck18``.

Constraints
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70
   :class: wrap

   * - ``name``
     - Keeps the draws with
   * - ``mass_order``
     - ``mass1`` at least ``mass2``.
   * - ``lambda_order``
     - ``lambda1`` at most ``lambda2``, for binaries with tidal parameters.

Waveform transforms
-------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65
   :class: wrap

   * - ``name``
     - Computes
   * - ``spherical_to_cartesian``
     - A spin's ``x``, ``y`` and ``z`` components from its size and
       direction, under the names set in ``x``, ``y``, ``z``, ``radial``,
       ``polar`` and ``azimuthal``.
   * - ``mass1_mass2_to_mchirp_q``
     - The chirp mass ``mchirp`` and the mass ratio ``q``.
   * - ``chirp_distance_to_distance``
     - The luminosity distance ``distance``, from ``chirp_distance`` and
       ``mchirp``.

A transform can use only parameters drawn or computed before it, so their
order matters.

Changing the prior
------------------

``sage init`` writes the whole prior into ``prior.yaml``, and the run reads
that file. To change the prior, edit it.

To change a range, change its ``min`` and ``max``. To draw ``mass1`` from 10
to 40 solar masses:

.. code-block:: yaml

   mass1:
     name: uniform
     min: 10.0
     max: 40.0

To draw a parameter from another distribution, change its ``name`` and give
the settings that distribution takes, from `Distributions`_. To draw the
chirp distance uniformly, instead of uniform in volume:

.. code-block:: yaml

   chirp_distance:
     name: uniform
     min: 130.0
     max: 350.0

``variable_params`` lists every parameter the priors draw. When you add a
prior, also add the parameters it draws to ``variable_params``; when you
remove a prior, also remove them.

Sage checks the prior each time it reads the run. If something is wrong, it
stops and says what: an unknown distribution or setting, a ``min`` that is
not a number, or ``variable_params`` that do not match the priors.
:doc:`mistakes` shows these messages, and :doc:`changing` shows more
examples.
