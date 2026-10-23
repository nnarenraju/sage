Synthetic Noise Generation
===========================

Not every job needs real detector strain. Unit tests, pipeline smoke tests, sanity
checks on a new architecture, and controlled experiments where you want to know the
exact spectrum of the data all call for noise you can *specify* rather than sample.

:func:`~sage.data.noise.sample_synthetic_noise` generates coloured Gaussian noise of a
given duration, with the ASD of your choice, on CPU or GPU, in the time domain or the
frequency domain. Its conventions follow LAL and PyCBC exactly, so the output is
directly comparable with noise produced by those libraries.

.. note::

   This is *simulated* noise. For real detector strain — the data used to train
   production models — see :doc:`sampling`.


The simplest case
-----------------

Give it a duration in seconds:

.. code-block:: python

    from sage.data.noise import sample_synthetic_noise

    x = sample_synthetic_noise(4.0)
    print(x.shape)     # torch.Size([8192])  -- 4 s at 2048 Hz

That is white Gaussian noise, returned as a ``torch.Tensor``. To colour it with a
detector spectrum, name a model:

.. code-block:: python

    x = sample_synthetic_noise(4.0, "aLIGOZeroDetHighPower")

The sample rate is 2048 Hz unless you say otherwise:

.. code-block:: python

    x = sample_synthetic_noise(4.0, sample_rate=4096.0)
    print(x.shape)     # torch.Size([16384])

Every other setting works the same way — an argument with a sensible default, which
you override when you need to. The full list is below.


If you don't know the model names
----------------------------------

Real noise curves have names like ``aLIGOZeroDetHighPower`` and
``AdVDesignSensitivityP1200087``. You do not need to know any of them.

Just name the detectors. Each one gets its own design sensitivity:

.. code-block:: python

    x = sample_synthetic_noise(4.0, detectors=["H1", "L1", "V1"])
    # H1 and L1 get the aLIGO curve, V1 gets the AdVirgo curve

(Writing ``asd="auto"`` does the same thing explicitly, and errors rather than
falling back to white if a name is not recognised.)

Or ask for a detector by its ordinary name:

.. code-block:: python

    x = sample_synthetic_noise(4.0, "LIGO")
    x = sample_synthetic_noise(4.0, ["LIGO", "Virgo"])

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Say this
     - and you get
   * - ``"LIGO"`` / ``"aLIGO"``
     - ``aLIGODesignSensitivityP1200087``
   * - ``"Virgo"`` / ``"AdV"``
     - ``AdVDesignSensitivityP1200087``
   * - ``"KAGRA"``
     - ``KAGRADesignSensitivityT1600593``
   * - ``"ET"`` / ``"Einstein Telescope"``
     - ``EinsteinTelescopeP1600143``
   * - ``"CE"`` / ``"Cosmic Explorer"``
     - ``CosmicExplorerP1600143``

These are shorthands for exact models, nothing more — the output is identical to
naming the model yourself. When you want a specific observing run or sensitivity
rather than the design curve, name it directly, as below.


Choosing an ASD
---------------

:func:`~sage.data.noise.available_asds` lists the analytic noise curves that ship with
LALSimulation — aLIGO, AdVirgo, KAGRA, Einstein Telescope and Cosmic Explorer, at
various observing runs and sensitivities:

.. code-block:: python

    from sage.data.noise import available_asds

    available_asds()            # 94 models
    available_asds("O4")        # ['AdVO4IntermediateT1800545', 'AdVO4T1800545', ...]
    available_asds("KAGRA")     # every KAGRA curve

.. figure:: /_static/noise_asd_models.png
   :alt: Analytic detector ASD models
   :align: center
   :width: 100%

   Three of the 94 available models. Any name from ``available_asds()`` can be passed
   straight to ``sample_synthetic_noise``.

You are not restricted to the built-in models. The ``asd`` argument also accepts:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Form
     - Meaning
   * - omitted (default)
     - Each named detector's own design curve; white noise if there is no
       recognised detector to infer one from.
   * - ``"white"`` / ``"flat"``
     - White noise, whatever the detectors are called.
   * - ``"auto"``
     - Each named detector's own design curve; errors if a name is unrecognised.
   * - ``"LIGO"``, ``"Virgo"``, …
     - A detector by its ordinary name (see above).
   * - ``"aLIGOZeroDetHighPower"``
     - An analytic model name, evaluated exactly on the output grid.
   * - ``(freqs, values)``
     - Your own ASD on your own frequency grid, interpolated onto the output grid.
   * - a 1-D array
     - Your own ASD, already on the output grid, used as is.
   * - a callable
     - Called as ``fn(freqs)`` and expected to return an ASD.
   * - a float
     - A flat ASD at that level.
   * - a **list** of any of these
     - One per detector; sets the detector axis if ``detectors`` is omitted.

For example, an ASD measured from real data:

.. code-block:: python

    import numpy as np

    freqs = np.loadtxt("my_asd.txt")[:, 0]
    values = np.loadtxt("my_asd.txt")[:, 1]

    x = sample_synthetic_noise(4.0, (freqs, values))

Pass ``is_psd=True`` if what you have is a PSD (:math:`\mathrm{Hz}^{-1}`) rather than
an ASD.


Batches and detectors
---------------------

Add ``batch`` and ``detectors`` to get the leading axes. Each appears only if you ask
for it:

.. code-block:: python

    sample_synthetic_noise(4.0).shape                              # (8192,)
    sample_synthetic_noise(4.0, batch=32).shape                    # (32, 8192)
    sample_synthetic_noise(4.0, detectors=2).shape                 # (2, 8192)
    sample_synthetic_noise(4.0, batch=32, detectors=2).shape       # (32, 2, 8192)

Detectors differ in one way that matters for noise: they have different sensitivity
curves. So ``asd`` and ``detectors`` between them decide both how many channels you
get and what each one looks like. Every combination is allowed:

**1. One ASD, two detectors** — both channels get the same curve, with independent
noise in each. This is the common case for an H1/L1 network:

.. code-block:: python

    x = sample_synthetic_noise(4.0, "aLIGOZeroDetHighPower", detectors=["H1", "L1"])
    print(x.shape)     # torch.Size([2, 8192])

**2. Two ASDs, two detectors** — one curve each, matched up in order:

.. code-block:: python

    x = sample_synthetic_noise(
        4.0,
        ["aLIGOZeroDetHighPower", "AdVO4T1800545"],
        detectors=["H1", "V1"],
    )

**3. Detectors only** — name real detectors and you get their noise. Nobody ever saw
white noise come out of H1, so saying nothing about the spectrum gives you each
detector's design curve rather than white:

.. code-block:: python

    x = sample_synthetic_noise(4.0, detectors=["H1", "L1"])   # (2, 8192), aLIGO
    x = sample_synthetic_noise(4.0, detectors=["H1", "V1"])   # aLIGO and AdVirgo

There has to be something to infer from. A plain count, or labels that are not
recognised detectors, gives white noise as before:

.. code-block:: python

    x = sample_synthetic_noise(4.0, detectors=2)              # white
    x = sample_synthetic_noise(4.0, detectors=["a", "b"])     # white

And ``asd="white"`` always means white, whatever the detectors are called:

.. code-block:: python

    x = sample_synthetic_noise(4.0, "white", detectors=["H1", "L1"])   # white

**4. ASDs only** — the list sets the detector axis by itself, so you need not state
the count twice. A single ASD leaves the axis off entirely:

.. code-block:: python

    x = sample_synthetic_noise(4.0, ["aLIGOZeroDetHighPower", "AdVO4T1800545"])
    print(x.shape)     # torch.Size([2, 8192])  -- two detectors, inferred

    x = sample_synthetic_noise(4.0, "aLIGOZeroDetHighPower")
    print(x.shape)     # torch.Size([8192])     -- no detector axis

Summarised:

.. list-table::
   :header-rows: 1
   :widths: 30 26 44

   * - ``asd``
     - ``detectors``
     - Result
   * - one model
     - ``["H1", "L1"]``
     - 2 channels, same curve, independent noise
   * - list of 2 models
     - ``["H1", "V1"]``
     - 2 channels, one curve each
   * - omitted
     - ``["H1", "L1"]``
     - 2 channels, each detector's own design curve
   * - omitted
     - ``2``
     - 2 channels of white noise — nothing to infer from
   * - ``"white"``
     - ``["H1", "L1"]``
     - 2 channels of white noise — asked for explicitly
   * - list of 2 models
     - omitted
     - 2 channels — the list sets the count
   * - one model
     - omitted
     - no detector axis at all
   * - ``"auto"``
     - ``["H1", "L1", "V1"]``
     - 3 channels, each detector's own design curve (same as omitting ``asd``)

If the ASD and the detector disagree, it is an error:

.. code-block:: python

    sample_synthetic_noise(4.0, "AdVO4T1800545", detectors=["H1", "L1"])
    # ValueError: detector 'H1' is a LIGO interferometer but was given the Virgo
    # curve 'AdVO4T1800545'.  Pass the model that matches the detector; or, if
    # the pairing is deliberate, drop the names and give a count (detectors=2)
    # or pass the ASD list on its own and let it set the detector axis.

Beyond the ASD, nothing about the noise is detector-specific — the names do not
change the output — so an H1 channel filled with AdVirgo noise is almost always a
misunderstanding rather than a choice, and is refused rather than silently produced.
If you do want a deliberately mismatched network, give a count instead of names.

The check only fires when it is certain: both the label and the model name have to be
recognisable and disagree. Integer ``detectors``, your own ASD arrays, and unfamiliar
labels all pass without comment. It also reads LALSimulation's deprecated aliases
correctly — ``aLIGOAdVO4T1800545`` is an AdVirgo curve despite the prefix.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Label
     - Expected model family
   * - ``H1``, ``L1``, ``I1``, ``A1``
     - aLIGO / A+ curves
   * - ``V1``
     - AdVirgo curves
   * - ``K1``
     - KAGRA curves
   * - ``E1``, ``E2``, ``E3``
     - Einstein Telescope curves
   * - ``C1``
     - Cosmic Explorer curves

.. note::

   Antenna patterns and light-travel time delays are **not** applied here, and should
   not be. They act on a gravitational wave arriving from a sky position; detector
   noise is independent instrumental noise in each interferometer. Projection belongs
   to the signal — see :class:`~sage.data.waveform.project.ConstantProjection`.


Time domain or frequency domain
-------------------------------

``domain="td"`` (the default) returns a real time series. ``domain="fd"`` returns the
complex spectrum in Sage's ``rfft(norm="forward")`` convention, ready to be combined
with a frequency-domain signal batch:

.. code-block:: python

    x = sample_synthetic_noise(4.0, domain="td")   # (8192,)   float32
    X = sample_synthetic_noise(4.0, domain="fd")   # (4097,)   complex64


Reproducibility
---------------

Pass ``seed`` to make a call reproducible. Without it, every call draws fresh entropy:

.. code-block:: python

    a = sample_synthetic_noise(4.0, seed=0)
    b = sample_synthetic_noise(4.0, seed=0)
    torch.equal(a, b)      # True


All options
-----------

Every argument, with its default. Only ``duration`` is required.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Argument
     - Default
     - What it does
   * - ``duration``
     - *required*
     - Length of the sample in **seconds**.
   * - ``asd``
     - omitted
     - What to colour the noise with. Left unset it follows the detectors — see
       the table of accepted forms above.
   * - ``sample_rate``
     - ``2048.0``
     - Samples per second. ``duration × sample_rate`` gives the length in samples.
   * - ``batch``
     - ``None``
     - Number of independent samples. ``None`` omits the axis.
   * - ``detectors``
     - ``None``
     - Detector count, or a list of names. ``None`` omits the axis, unless ``asd``
       is a list, which sets it. Recognised names also select the default ASD.
   * - ``domain``
     - ``"td"``
     - ``"td"`` for a real time series, ``"fd"`` for the complex
       ``rfft(norm="forward")`` spectrum.
   * - ``seed``
     - ``None``
     - Any integer makes the call reproducible. ``None`` draws fresh entropy each
       call.
   * - ``unit_psd``
     - ``True``
     - ``True`` gives white noise a flat unit one-sided PSD
       (:math:`\sigma = \sqrt{f_s/2}`); ``False`` gives unit variance. Ignored
       when an ASD is supplied — the output is then in the ASD's own units.
   * - ``low_frequency_cutoff``
     - ``None``
     - Zero the ASD below this frequency, in Hz. For a named model this defaults
       to 10 Hz — see the note below.
   * - ``filter_duration``
     - ``None``
     - Inverse-spectrum-truncation length in seconds. ``None`` applies no
       truncation and colours with the exact ASD.
   * - ``threads``
     - ``1``
     - Worker threads for the random draw. Speed only — never changes the
       numbers. See :ref:`speed <synthetic-noise-speed>`.
   * - ``numpy``
     - ``False``
     - Return a NumPy array instead of a ``torch.Tensor``.
   * - ``device``
     - ``None``
     - Where to generate, e.g. ``"cuda"``. ``None`` means CPU.
   * - ``dtype``
     - ``torch.float32``
     - The real dtype. ``domain="fd"`` returns the matching complex type. Leave
       it at float32 — a float64 draw is about six times slower and an ASD needs
       nothing finer.

.. note::

   **On** ``low_frequency_cutoff``. The analytic models rise almost vertically into
   the seismic wall at low frequency, so evaluating one down to 0 Hz buries the
   sample under low-frequency power some sixteen orders of magnitude above the
   mid-band. When you name a model and give no cutoff, 10 Hz is applied for you. An
   ASD you supply yourself is never touched — your array is used exactly as given.


Verifying the output
--------------------

The generated noise reproduces the ASD you asked for. Measuring the spectrum of a
batch and comparing it against the requested model agrees to better than a percent
across the band:

.. figure:: /_static/noise_synthetic_example.png
   :alt: Generated strain and its measured spectrum
   :align: center
   :width: 100%

   *Top*: one second of the generated strain for ``aLIGOZeroDetHighPower``.
   *Bottom*: the spectrum measured from 256 generated samples, against the model that
   was requested.


.. _synthetic-noise-speed:

Speed
-----

Noise generation is bottlenecked by the random draw, not by the FFTs. Measured on a
96-core AMD EPYC 9654, for two detectors and 10-second samples at 2048 Hz, batch 256.

Two rates are quoted, because "sample" is ambiguous:

* **samples / s** — complete examples per second. One sample here is one
  two-detector, 10-second example: a ``(2, 20480)`` array. This is the number you
  care about when you are counting training examples.
* **M points / s** — millions of individual numbers per second, counting every
  detector at every time step (:math:`\text{batch} \times \text{detectors} \times
  \text{time steps}`). This is raw throughput, useful for comparing against a
  different duration or detector count.

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Configuration
     - ms / batch
     - samples / s
     - M points / s
   * - White, time domain
     - 68
     - 3 770
     - 154
   * - Coloured, time domain
     - 129
     - 1 980
     - 80
   * - Coloured, ``threads=8``
     - 58
     - 4 420
     - 181
   * - Coloured, ``threads=16``
     - 52
     - 4 970
     - 204
   * - Coloured, ``threads=32``
     - 49
     - 5 190
     - 213

PyTorch's CPU normal generator is single-threaded — it takes the same time on one core
as on ninety-six — and on a large batch it is roughly two thirds of the call. The
``threads`` argument splits the draw across worker threads to recover that:

.. code-block:: python

    x = sample_synthetic_noise(
        10.0, "aLIGOZeroDetHighPower",
        batch=256, detectors=2, threads=16,
    )

.. important::

   ``threads`` changes speed and **never** the numbers. Chunk boundaries and per-chunk
   seeds are fixed by the batch shape, not by the thread count or the machine, so a
   given seed produces bit-identical output at any thread count.

It defaults to ``1``. Turn it up for standalone generation; leave it alone inside a
training loop or a DataLoader worker, where a thread pool would compete with the rest
of the process for cores. It is ignored on CUDA, whose generator is already parallel.

Two further notes on performance:

* **Generate where you will use it.** Pass ``device="cuda"`` to build the batch on
  the GPU directly, rather than assembling it on the CPU and copying it across.
* **Keep ``dtype`` at float32.** It is the default. A float64 draw is about six times
  slower, and an ASD needs nothing beyond single precision.


Lower-level building blocks
---------------------------

:func:`~sage.data.noise.sample_synthetic_noise` covers the common case. The module
underneath exposes the individual pieces for anything else:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Function
     - Purpose
   * - ``white_noise_td`` / ``white_noise_fd``
     - White noise of a given shape, in either domain.
   * - ``coloured_noise_td``
     - Colour a batch with an ASD, padding and cropping to avoid edge transients.
   * - ``colour_fd``
     - Multiply a unit-PSD spectrum by an ASD.
   * - ``resolve_asd``
     - Turn any ASD specification into a tensor on a target frequency grid.
   * - ``white_series`` / ``coloured_series``
     - Block-addressable noise over a GPS interval: overlapping requests agree
       exactly, and any interval can be generated without materialising the ones
       before it. ``legacy_pycbc=True`` reproduces
       ``pycbc.noise.reproduceable.normal`` bit for bit.
   * - ``feather``
     - LAL's power-complementary crossfade, for stitching long coloured streams.

There is also :class:`~sage.data.noise.WhiteGaussianNoiseSampler`, a batch sampler that
mirrors the :class:`~sage.data.noise.MemmapNoiseSampler` API so it can be dropped into
a training loop wherever real noise is used. It is index-addressable: the batch at step
*n* depends only on ``(seed, n)``, so a run resumed mid-stream reproduces the same
noise.


Conventions
-----------

The module follows the LAL/PyCBC one-sided PSD convention throughout, where
:math:`S(f)` is in :math:`\mathrm{Hz}^{-1}` and :math:`\mathrm{ASD} = \sqrt{S}`.

A real series sampled at :math:`f_s` whose one-sided PSD is flat at :math:`S` has
variance

.. math::

   \mathrm{var}(x) = S \, f_s / 2

so *unit-PSD* white noise has :math:`\sigma = \sqrt{f_s/2}`, **not** 1. This is the
default (``unit_psd=True``), and the reason for it is that colouring then becomes a
bare multiplication by the ASD with no leftover factors anywhere in the chain. Pass
``unit_psd=False`` for zero-mean, unit-variance output instead; it is ignored when an
ASD is supplied, since the output is then in the ASD's own units.

In the frequency domain, with Sage's ``rfft(norm="forward")``:

.. math::

   \mathbb{E}|X_k|^2 = S(f_k) \, \Delta f / 2 \qquad \text{for every } k

with :math:`X_k` **real** at DC and Nyquist. A real inverse FFT discards the imaginary
part of those two bins, so drawing them complex — as LAL and ``pycbc.noise.gaussian``
both do — leaves them at half power. This module draws them real.
