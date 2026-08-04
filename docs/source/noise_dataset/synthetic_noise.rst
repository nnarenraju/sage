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

That is white Gaussian noise. To colour it with a detector spectrum, name a model:

.. code-block:: python

    x = sample_synthetic_noise(4.0, "aLIGOZeroDetHighPower")

The sample rate defaults to the registered data config's ``sample_rate`` (2048 Hz if
no config is registered), and the output is a ``torch.Tensor``.


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
   * - ``None`` (default)
     - White noise — no colouring.
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

``detectors`` accepts a count or a list of names — only the length is used:

.. code-block:: python

    x = sample_synthetic_noise(4.0, detectors=["H1", "L1", "V1"])   # (3, 8192)

Detectors differ in one way that matters for noise: they have different sensitivity
curves. Pass a **list** of ASDs, one per detector, to give each its own:

.. code-block:: python

    x = sample_synthetic_noise(
        4.0,
        ["aLIGOZeroDetHighPower", "AdVO4T1800545"],
        detectors=["H1", "V1"],
        batch=32,
    )

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


Speed
-----

Noise generation is bottlenecked by the random draw, not by the FFTs. Measured on a
96-core AMD EPYC 9654, for two detectors and 10-second samples at 2048 Hz, batch 256
(one "sample" being one two-detector 10-second example):

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Configuration
     - ms / batch
     - samples / s
     - Msamples / s
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

* **Generate where you will use it.** ``device`` defaults to the registered config's
  ``device``, so a configured run generates straight onto the GPU instead of building
  the batch on the CPU and copying it across.
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
