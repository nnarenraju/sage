Configuration Reference
========================

A Sage production run is controlled by two files that live alongside the run script:

- **``config.py``** — ``RunCFG`` and ``DataCFG``: training loop settings, paths to data files.
- **``gwconfig.yaml``** — parameter priors, constraints, and waveform transforms for the signal sampler.

The run script (``train.py``) registers these configs once at startup via
``register_configs`` and then reads them anywhere in the codebase via
``get_cfg()`` / ``get_data_cfg()``.

The files shown here are taken directly from the O3b production run under ``runs/o3b/``.

----

Run configuration (``config.py``)
------------------------------------

``RunCFG``
~~~~~~~~~~~

Controls the training loop and hardware settings.

.. code-block:: python

   class O3bCFG:
       export_dir            = "./run_export"
       batch_size            = 128
       device                = "cuda:0"
       dtype                 = torch.float32
       detectors             = ["H1", "L1"]
       do_point_estimate     = ["tc", "mchirp"]
       autocast              = True
       class_balance         = 0.5
       clip_norm             = 1.0
       num_epochs            = 80
       training_iterations   = int(2_000_000 / batch_size)
       validation_iterations = int(200_000  / batch_size)

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``export_dir``
     - ``"./run_export"``
     - Directory where checkpoints, loss logs, and exported artefacts are written.
   * - ``batch_size``
     - ``128``
     - Number of samples per training step. Halving this roughly halves peak VRAM usage.
   * - ``device``
     - ``"cuda:0"``
     - PyTorch device string. Use ``"cpu"`` for debugging without a GPU.
   * - ``dtype``
     - ``torch.float32``
     - Floating-point precision for model weights and activations. ``float32`` is the
       baseline; ``autocast`` handles the mixed-precision forward pass separately.
   * - ``detectors``
     - ``["H1", "L1"]``
     - Active detector network. Each entry must have a corresponding noise file in
       ``DataCFG.training_noise_files``.
   * - ``do_point_estimate``
     - ``["tc", "mchirp"]``
     - Parameters for which the model produces a heteroscedastic regression output
       (mean + uncertainty) in addition to the detection score. Must be a subset of
       the parameters listed in ``gwconfig.yaml``.
   * - ``autocast``
     - ``True``
     - Enables ``torch.amp`` automatic mixed-precision for the forward pass, cutting
       activation memory roughly in half. Always pair with a ``GradScaler``.
   * - ``class_balance``
     - ``0.5``
     - Fraction of each batch drawn from the signal class. ``0.5`` means equal numbers
       of signal and noise samples per batch.
   * - ``clip_norm``
     - ``1.0``
     - Maximum L2 norm for gradient clipping (passed to ``torch.nn.utils.clip_grad_norm_``).
       Set to ``None`` to disable.
   * - ``num_epochs``
     - ``80``
     - Total number of training epochs.
   * - ``training_iterations``
     - ``int(2_000_000 / batch_size)``
     - Number of batches per training epoch, expressed as total samples divided by
       batch size. With ``batch_size=128`` this is ~15 625 steps per epoch,
       corresponding to 2 million samples seen per epoch.
   * - ``validation_iterations``
     - ``int(200_000 / batch_size)``
     - Number of batches per validation pass. Validation runs on epoch 0 and then
       every 5 epochs.

``DataCFG``
~~~~~~~~~~~~

Controls data paths and signal processing parameters shared across the entire pipeline.

.. code-block:: python

   class O3bDataCFG:
       data_dir                  = "/path/to/data_dir"
       training_noise_files      = [
           "/path/to/data_H1_O3b.bin",
           "/path/to/data_L1_O3b.bin",
       ]
       validation_noise_files    = [
           "/path/to/data_H1_O3a.bin",
           "/path/to/data_L1_O3a.bin",
       ]
       sample_rate               = 2048.0   # Hz
       noise_low_frequency_cutoff = 15.0   # Hz
       signal_low_frequency_cutoff = 20.0  # Hz
       sample_length_in_s        = 12.0    # seconds
       padding_length_in_s       = 2.0     # seconds

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Field
     - Default (O3b)
     - Description
   * - ``data_dir``
     - *(path)*
     - Parent directory for all intermediate data products (PSDs, processed segments).
   * - ``training_noise_files``
     - O3b H1 + L1 ``.bin``
     - Monolithic noise files produced by the data priming step, one per detector,
       used for the training split. Order must match ``detectors`` in ``RunCFG``.
   * - ``validation_noise_files``
     - O3a H1 + L1 ``.bin``
     - Noise files for the validation split. Using a different observing run here
       (O3a vs O3b) provides an out-of-distribution test of noise generalisation.
   * - ``sample_rate``
     - ``2048.0``
     - Target sample rate in Hz after downsampling. All time-domain operations use
       this rate.
   * - ``noise_low_frequency_cutoff``
     - ``15.0``
     - Low-frequency cutoff applied when downloading and whitening noise data, in Hz.
   * - ``signal_low_frequency_cutoff``
     - ``20.0``
     - Low-frequency cutoff applied during waveform generation, in Hz.
   * - ``sample_length_in_s``
     - ``12.0``
     - Total duration of each data sample fed to the network, in seconds.
   * - ``padding_length_in_s``
     - ``2.0``
     - Extra data fetched at both edges of each noise window before whitening.
       Whitening corrupts the edges of the time series; the padding region absorbs
       this corruption and is trimmed away, leaving a clean ``sample_length_in_s``
       window for the network.

----

Signal prior configuration (``gwconfig.yaml``)
------------------------------------------------

``gwconfig.yaml`` is read by ``read_from_config`` and passed to the signal sampler.
It defines which waveform parameters are sampled, their prior distributions,
physical constraints, and transforms applied before waveform generation.

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

``variable_params``
~~~~~~~~~~~~~~~~~~~~

Lists every parameter that is sampled per injection. All entries must have a
corresponding prior defined under ``priors``.

``priors``
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Prior
     - Notes
   * - ``injection_time``
     - ``uniform``  [1238166018, 1253977218]
     - GPS start time of the noise window. Range spans the O3b observing period
       (November 2019 – March 2020).
   * - ``tc``
     - ``uniform``  [11.0, 11.2]
     - Coalescence time in seconds, measured within the 12 s sample window.
       Placing the merger near the end leaves a long inspiral visible to the network.
   * - ``mass1``
     - ``uniform``  [7, 50] M☉
     - Primary component mass. The ``mass_order`` constraint enforces ``mass1 ≥ mass2``.
   * - ``mass2``
     - ``uniform``  [7, 50] M☉
     - Secondary component mass.
   * - ``spin1_a``
     - ``uniform``  [0, 0.99]
     - Spin magnitude of the primary, in units of the maximum Kerr spin.
   * - ``spin1_orientation``
     - ``uniform_solidangle``
     - Spin tilt and azimuthal angle sampled uniformly on the unit sphere via
       ``spin1_polar`` and ``spin1_azimuthal``.
   * - ``spin2_a``
     - ``uniform``  [0, 0.99]
     - Spin magnitude of the secondary.
   * - ``spin2_orientation``
     - ``uniform_solidangle``
     - Spin orientation of the secondary, sampled as above.
   * - ``sky``
     - ``uniform_sky``
     - Right ascension and declination sampled uniformly over the sphere.
   * - ``inclination``
     - ``sin_angle``
     - Binary inclination angle; the ``sin_angle`` prior is isotropic over the sphere.
   * - ``coa_phase``
     - ``uniform_angle``
     - Coalescence phase, uniform on [0, 2π).
   * - ``polarization``
     - ``uniform_angle``
     - Gravitational-wave polarisation angle, uniform on [0, 2π).
   * - ``chirp_distance``
     - ``uniform_radius``  [130, 350] Mpc
     - Distance parameterised by the chirp-mass–weighted luminosity distance.
       ``uniform_radius`` samples uniformly in volume (∝ r²). The ``distance``
       transform converts this to a luminosity distance for waveform generation.

``constraints``
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   constraints:
     - name: mass_order

``mass_order`` rejects samples where ``mass1 < mass2``, enforcing the convention
that the primary is always the heavier component.

``waveform_transforms``
~~~~~~~~~~~~~~~~~~~~~~~~

Applied after sampling, before waveform generation.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Transform
     - Effect
   * - ``spin1_cartesian``
     - Converts ``(spin1_a, spin1_polar, spin1_azimuthal)`` → ``(spin1x, spin1y, spin1z)``
       for the waveform generator.
   * - ``spin2_cartesian``
     - Same conversion for the secondary spin.
   * - ``mass_params``
     - Computes ``mchirp`` and mass ratio ``q`` from ``mass1`` and ``mass2``.
   * - ``distance``
     - Converts ``chirp_distance`` to luminosity distance.

----

Training graph (``train.py``)
-------------------------------

Signal sampler
~~~~~~~~~~~~~~~

.. code-block:: python

   param_sampler  = read_from_config("./gwconfig.yaml", seed=150914)
   waveform_project = ConstantProjection()
   snrscaler      = OptimalSNRRescaler(HalfNorm(scale=4.0, loc=5.0, seed=150914))
   signal_sampler = IMRPhenomPv2(param_sampler, waveform_project, augment=snrscaler)

The seeds used here (``150914``, ``170817``) match the GPS dates of GW150914 and
GW170817 — the first binary black hole and binary neutron star detections respectively.
Validation uses the alternate seed to guarantee independent sampling.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Role
   * - ``read_from_config``
     - Parses ``gwconfig.yaml`` and returns a parameter sampler that draws from the
       defined priors on each call.
   * - ``ConstantProjection``
     - Projects the waveform onto the detector network using antenna pattern factors that
       are held fixed over the duration of the signal. This is accurate for CBC signals,
       which are short enough that the detector orientation relative to the source does
       not change appreciably due to Earth's rotation.
   * - ``HalfNorm(scale=4.0, loc=5.0)``
     - Samples target optimal SNR values from a half-normal distribution with
       location 5 and scale 4 (most injections fall between SNR 5 and ~15).
   * - ``OptimalSNRRescaler``
     - Rescales each waveform amplitude so its optimal SNR matches the sampled target.
   * - ``IMRPhenomPv2``
     - Wraps all of the above; calling ``signal_sampler()`` returns a ready-to-use
       ``(signal_data, signal_targets)`` tuple for one batch.

Noise sampler
~~~~~~~~~~~~~~

.. code-block:: python

   recolour = RecolourPostprocess(
       p_recolour=0.37,
       recolour_dataset_dir="/path/to/o3a",
   )
   noise_sampler = MemmapNoiseSampler(postprocess_fn=recolour, prefetch=8, seed=150914)

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``p_recolour``
     - Probability that a noise window is recoloured with a PSD drawn from a different
       segment. ``0.37`` means roughly one in three noise samples is augmented this way.
       This value was set from a coarse ablation and should be tuned to the diversity of
       the recolouring PSD library: a diverse library can tolerate a higher value; too
       high a value is harmful because recolouring is imperfect and the training
       distribution can diverge from the test data, which has no recolouring applied.
   * - ``recolour_dataset_dir``
     - Directory containing the O3a data used as the recolouring PSD library.
   * - ``prefetch``
     - Number of batches prefetched to the GPU ahead of training. Increase on
       disk-bound systems; decrease if VRAM is tight.

Preprocessor
~~~~~~~~~~~~~

.. code-block:: python

   whitener  = FiducialWhitening()
   binning   = DyadicPyramidBinning(bounds)
   mrsampler = MultirateSampler(binning_method=binning)
   processor = Preprocessor([whitener, mrsampler])

Applies whitening followed by multirate compression. The binning boundaries are
derived from the parameter prior ``bounds`` so that the time-resolution allocation
matches the signal's frequency evolution across the full prior volume.

.. note::

   Inverse Spectrum Truncation (:term:`IST`) is intentionally disabled during PSD
   estimation (``apply_inverse_spectrum_truncation=False`` in ``dataset.py``). IST
   smooths sharp spectral lines by windowing the inverse PSD in the time domain, but
   this also removes fine spectral features. Preserving those features keeps the PSD
   library more varied, which benefits recolouring augmentation during training.

----

Model and optimisation (``train.py``)
---------------------------------------

.. code-block:: python

   model = MSCNN1D_2DResNetCBAM_Heteroscedastic(
       frontend_filters  = 32,
       frontend_kernel   = 64,
       backend_resnet_size = 50,
       norm_type         = "instancenorm",
   ).to(dtype=cfg.dtype, device=cfg.device, memory_format=torch.channels_last)

   model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)

   loss_function = BCEWithPEsigmaLoss(regression_weight=0.005, coupling_weight=0.005)
   optimiser     = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6, fused=True)
   scheduler     = CosineAnnealingWarmRestarts(optimiser, T_0=5, T_mult=2, eta_min=1e-6)
   scaler        = torch.amp.GradScaler(cfg.device, enabled=cfg.autocast)

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Notes
   * - ``frontend_filters=32``
     - Number of convolutional filters in the multi-scale frontend.
   * - ``frontend_kernel=64``
     - Largest kernel size in the frontend; controls the longest time-scale feature
       the network can detect at full resolution.
   * - ``backend_resnet_size=50``
     - ResNet depth for the 2-D backend (50, 101, or 152). ResNet-50 is the
       recommended default; larger sizes increase VRAM usage substantially.
   * - ``norm_type="instancenorm"``
     - Normalisation layer type. ``"instancenorm"`` is stable with small batch sizes
       and works well under ``autocast``.
   * - ``torch.compile(..., mode="max-autotune")``
     - Kernel auto-tuning; the first epoch is slow but steady-state throughput improves
       significantly. ``fullgraph=True`` requires no graph breaks in the model.
   * - ``BCEWithPEsigmaLoss``
     - Combined binary cross-entropy (detection) and heteroscedastic regression
       (parameter estimation) loss.
   * - ``regression_weight=0.005``
     - Relative weight of the regression term in the total loss.
   * - ``coupling_weight=0.005``
     - Relative weight of the uncertainty coupling term.
   * - ``lr=2e-4``
     - Initial learning rate. Reduce if training shows NaN losses early on.
   * - ``CosineAnnealingWarmRestarts(T_0=5, T_mult=2)``
     - Learning rate schedule that restarts every ``T_0`` epochs, doubling the period
       on each restart (5, 10, 20, 40 … epochs).
