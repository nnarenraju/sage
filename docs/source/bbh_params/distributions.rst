Available Distributions
========================

All distributions live in :mod:`sage.data.waveform.distributions` and are referenced
by name in ``gwconfig.yaml``. Every distribution exposes a ``sample(shape, device,
dtype, generator)`` interface that runs entirely on the target device.

``uniform``
-----------

**Class:** :class:`~sage.data.waveform.distributions.uniform.Uniform`

Draws samples uniformly from a closed interval ``[min, max]``.
Used for masses, spin magnitudes, coalescence time, and GPS injection time.

.. code-block:: yaml

    mass1:
      name: uniform
      min: 7.0
      max: 50.0

**Physics:** A uniform mass prior is maximally agnostic about the mass distribution. It
overrepresents high masses relative to the astrophysical population but ensures the
network is exposed to the full range without bias toward any mass scale.

``uniform_angle``
-----------------

**Class:** :class:`~sage.data.waveform.distributions.angular.UniformAngle`

Draws angles uniformly from ``[0, 2π)`` with optional wrapping. Used for coalescence
phase and polarisation angle, which are both cyclic.

.. code-block:: yaml

    coa_phase:
      name: uniform_angle

    polarization:
      name: uniform_angle

**Physics:** The coalescence phase and polarisation angle have no preferred value for
a population of unresolved sources, so a uniform prior over the full cycle is the
natural choice.

``sin_angle``
-------------

**Class:** :class:`~sage.data.waveform.distributions.angular.SinAngle`

Draws polar angles from the distribution proportional to ``sin(θ)`` over ``[0, π]``
via the inverse-CDF method in ``cos(θ)`` space. Used for the binary inclination angle.

.. code-block:: yaml

    inclination:
      name: sin_angle

**Physics:** A uniform distribution over a sphere has a polar-angle marginal
proportional to ``sin(θ)``. Using this prior for inclination ensures that face-on and
edge-on orientations occur with their correct relative frequency for a population of
isotropically oriented binaries.

``uniform_sky``
---------------

**Class:** :class:`~sage.data.waveform.distributions.sky.UniformSky`

Jointly samples right ascension and declination so that the resulting sky positions are
uniform over the 2-sphere. RA is drawn uniformly from ``[0, 2π)``; declination is drawn
from a ``cos(δ)``-weighted distribution via ``CosAngle``.

.. code-block:: yaml

    sky:
      name: uniform_sky
      ra: ra
      dec: dec

**Physics:** An isotropic source population has sky positions that are uniform in solid
angle, which means uniform in RA and uniform in ``sin(dec)``.  Naïvely drawing dec from
a flat distribution would oversample the poles.

``uniform_solidangle``
-----------------------

**Class:** :class:`~sage.data.waveform.distributions.angular.UniformSolidAngle`

Jointly samples a polar angle (via ``sin_angle``) and an azimuthal angle
(via ``uniform_angle``) to produce orientation vectors that are isotropically
distributed on the sphere. Used for 3-D spin orientations.

.. code-block:: yaml

    spin1_orientation:
      name: uniform_solidangle
      polar-angle: spin1_polar
      azimuthal-angle: spin1_azimuthal

    spin2_orientation:
      name: uniform_solidangle
      polar-angle: spin2_polar
      azimuthal-angle: spin2_azimuthal

**Physics:** Spin orientations in the absence of tidal alignment are expected to be
isotropically distributed. Uniform solid-angle sampling (rather than independent
uniform in both angles) correctly implements this.

``uniform_radius``
------------------

**Class:** :class:`~sage.data.waveform.distributions.powerlaw.UniformRadius`

Draws distances from a distribution proportional to ``r²`` over ``[min, max]``, which
corresponds to uniform spatial number density in a Euclidean volume. Used for chirp
distance as a proxy for luminosity distance.

.. code-block:: yaml

    chirp_distance:
      name: uniform_radius
      min: 130.0
      max: 350.0

**Physics:** For a volume-limited survey of uniformly distributed sources, the number
of sources within distance ``r`` grows as ``r³``, so ``p(r) ∝ r²``. Sampling chirp
distance rather than luminosity distance flattens the effective SNR distribution,
giving more training examples near the detection threshold.
