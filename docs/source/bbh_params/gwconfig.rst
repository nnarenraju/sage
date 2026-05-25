Writing gwconfig.yaml
======================

The prior configuration file is a YAML document with four top-level sections:
``variable_params``, ``priors``, ``constraints``, and ``waveform_transforms``.
Together they fully specify the parameter space from which every training waveform
is drawn.

Complete reference example
---------------------------

.. code-block:: yaml

    variable_params:
      - mass1
      - mass2
      - ra
      - dec
      - inclination
      - coa_phase
      - polarization
      - chirp_distance
      - spin1_a
      - spin1_azimuthal
      - spin1_polar
      - spin2_a
      - spin2_azimuthal
      - spin2_polar
      - injection_time
      - tc

    priors:

      injection_time:
        name: uniform
        min: 1238166018   # GPS start of O3a
        max: 1253977218   # GPS end of O3a

      tc:
        name: uniform
        min: 11.0         # earliest coalescence time (s) within the segment
        max: 11.2

      mass1:
        name: uniform
        min: 7.0
        max: 50.0

      mass2:
        name: uniform
        min: 7.0
        max: 50.0

      spin1_a:
        name: uniform
        min: 0.0
        max: 0.99

      spin1_orientation:
        name: uniform_solidangle
        polar-angle: spin1_polar
        azimuthal-angle: spin1_azimuthal

      spin2_a:
        name: uniform
        min: 0.0
        max: 0.99

      spin2_orientation:
        name: uniform_solidangle
        polar-angle: spin2_polar
        azimuthal-angle: spin2_azimuthal

      sky:
        name: uniform_sky
        ra: ra
        dec: dec

      inclination:
        name: sin_angle

      coa_phase:
        name: uniform_angle

      polarization:
        name: uniform_angle

      chirp_distance:
        name: uniform_radius
        min: 130.0
        max: 350.0

    constraints:
      - name: mass_order

    waveform_transforms:

      spin1_cartesian:
        name: spherical_to_cartesian
        x: spin1x
        y: spin1y
        z: spin1z
        radial: spin1_a
        polar: spin1_polar
        azimuthal: spin1_azimuthal

      spin2_cartesian:
        name: spherical_to_cartesian
        x: spin2x
        y: spin2y
        z: spin2z
        radial: spin2_a
        polar: spin2_polar
        azimuthal: spin2_azimuthal

      mass_params:
        name: mass1_mass2_to_mchirp_q

      distance:
        name: chirp_distance_to_distance

Section reference
------------------

``variable_params``
~~~~~~~~~~~~~~~~~~~~

A flat list of every raw parameter name that is sampled from a prior. The order here
does not affect sampling; the final output tensor uses alphabetically sorted names.
Parameters produced only by transforms (``spin1x``, ``mchirp``, ``distance``, …)
should **not** appear here — they are added automatically.

``priors``
~~~~~~~~~~~

Each entry maps a prior name to a distribution definition. The ``name`` field selects
the distribution class (see :doc:`distributions`). All other fields are passed as
keyword arguments to the class constructor.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Effect
   * - ``name``
     - Distribution type. One of ``uniform``, ``uniform_angle``, ``sin_angle``,
       ``uniform_sky``, ``uniform_solidangle``, ``uniform_radius``.
   * - ``min`` / ``max``
     - Bounds for ``uniform`` and ``uniform_radius``.
   * - ``polar-angle`` / ``azimuthal-angle``
     - Output parameter names for ``uniform_solidangle`` joint sampler.
   * - ``ra`` / ``dec``
     - Output parameter names for ``uniform_sky`` joint sampler.

Changing the mass range
^^^^^^^^^^^^^^^^^^^^^^^

To restrict the search to neutron-star–black-hole (NSBH) binaries:

.. code-block:: yaml

    mass1:
      name: uniform
      min: 5.0
      max: 20.0

    mass2:
      name: uniform
      min: 1.0
      max: 3.0

To search only equal-mass systems near the pair-instability gap:

.. code-block:: yaml

    mass1:
      name: uniform
      min: 40.0
      max: 80.0

    mass2:
      name: uniform
      min: 40.0
      max: 80.0

Changing the distance range
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Narrowing the chirp distance range increases the fraction of high-SNR events in
the training set. Widening it extends coverage toward the detection horizon but
produces more threshold-SNR examples. The ``uniform_radius`` distribution
automatically assigns the correct :math:`r^2` weight.

.. code-block:: yaml

    chirp_distance:
      name: uniform_radius
      min: 100.0   # Mpc  — closer / louder
      max: 600.0   # Mpc  — further / quieter

``constraints``
~~~~~~~~~~~~~~~~

Named Boolean filters applied via **rejection sampling** after the base parameters
are drawn. Currently one named constraint is available:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Name
     - Effect
   * - ``mass_order``
     - Enforces :math:`m_1 \geq m_2` by swapping components when the sampled
       masses violate the ordering. This is a deterministic projection, not true
       rejection, so it does not waste samples.

To remove the mass-ordering constraint (useful when the waveform model already handles
it):

.. code-block:: yaml

    constraints: []

``waveform_transforms``
~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic reparametrisations applied to the sampled base parameters before the
tensor is returned. Three transforms are available:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``name``
     - Description
   * - ``spherical_to_cartesian``
     - Converts spherical spin components ``(a, polar, azimuthal)`` to Cartesian
       ``(x, y, z)`` via :math:`x = a \sin\theta \cos\phi`,
       :math:`y = a \sin\theta \sin\phi`, :math:`z = a \cos\theta`.
       Required by IMRPhenomPv2.
   * - ``mass1_mass2_to_mchirp_q``
     - Computes chirp mass :math:`\mathcal{M}_c` and mass ratio :math:`q = m_1/m_2`
       from the component masses. Added to the output tensor as ``mchirp`` and ``q``.
   * - ``chirp_distance_to_distance``
     - Converts chirp distance to luminosity distance using the chirp mass:
       :math:`d_L = d_\text{chirp} \cdot \left(\mathcal{M}_c / 1.2\,M_\odot\right)^{5/6}`.
       Added as ``distance``.

Transforms that are not needed can simply be omitted from the ``waveform_transforms``
block.
