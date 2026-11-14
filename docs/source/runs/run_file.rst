The Run File
============

``run.yaml`` holds every setting of a run. It is plain YAML: sections, each
holding settings and their values.

A complete run file
-------------------

This run file is made by ``sage init my_run --preset three_class``. It has
every section and every optional setting:

.. literalinclude:: three_class_run.yaml
   :language: yaml

Sections
--------

.. list-table::
   :header-rows: 1
   :widths: 22 78
   :class: wrap

   * - Section
     - What it sets
   * - ``data``
     - The detectors, and the windows of data the network sees.
   * - ``signal``
     - The simulated signals: their waveform and loudness, and the
       parameters the network estimates. Their prior is in ``prior.yaml``.
   * - ``noise``
     - The detector noise the signals are added to, for training and for
       validation.
   * - ``preprocess``
     - What is done to each window before the network sees it.
   * - ``model``
     - The network.
   * - ``loss``
     - What training minimises.
   * - ``optimiser``
     - How the network's weights are updated.
   * - ``schedule``
     - How the learning rate changes during training.
   * - ``train``
     - How long training lasts, the batch size and the device.
   * - ``validation``
     - How often, and on how much data, the network is checked.
   * - ``mining``
     - Hard-noise mining: finding noise the network mistakes for a signal,
       and training on it more.
   * - ``ema``
     - Averaging the network's weights over training.

Some sections and settings are optional. A run without them does without
what they add:

.. list-table::
   :header-rows: 1
   :widths: 35 65
   :class: wrap

   * - Left out
     - The run then
   * - ``schedule``
     - keeps a constant learning rate.
   * - ``mining``
     - mines nothing.
   * - ``ema``
     - averages nothing.
   * - ``noise.recolour``
     - trains on the noise as it is.
   * - ``signal.non_astrophysical``
     - has two classes, signal and noise.

Every setting, with its meaning and allowed values, is listed in
:doc:`../configuration`. :doc:`changing` shows common changes.

Choosing parts by name
----------------------

Some settings choose a part of Sage by its ``name``, followed by that part's
own parameters:

.. code-block:: yaml

   model:
     name: MSCNN1Datt_2DResNetCBAM
     frontend_filters: 32
     frontend_kernel: 64
     backend_resnet_size: 50
     norm_type: instancenorm
     dropout: 0.0
     use_blurpool: true
     use_resnet_cd: true

``sage list`` shows the parts to choose from, each with its parameters and
their defaults:

.. code-block:: console

   $ sage list --help
   usage: sage list [-h] thing

   List what a run file can name, or what is downloaded.

   positional arguments:
     thing       presets, priors, models, losses, waveforms, optimisers, schedules, asds, data

   options:
     -h, --help  show this help message and exit

.. list-table::
   :header-rows: 1
   :widths: 40 60
   :class: wrap

   * - Setting
     - Choices from
   * - ``model``
     - ``sage list models``
   * - ``loss``
     - ``sage list losses``
   * - ``signal.waveform``
     - ``sage list waveforms``
   * - ``optimiser``
     - ``sage list optimisers``
   * - ``schedule``
     - ``sage list schedules``
   * - ``asd`` of synthetic noise
     - ``sage list asds``

``sage list presets`` is on :doc:`presets`, and ``sage list data`` shows
what is downloaded.

.. code-block:: console

   $ sage list models
   Models (parameters with their defaults):
     MSCNN1D_2DResNetCBAM: frontend_filters=32, frontend_kernel=64, backend_resnet_size=50, norm_type='instancenorm', dropout=0.0, use_blurpool=True, use_resnet_cd=True
     MSCNN1Datt_2DResNetCBAM: frontend_filters=32, frontend_kernel=64, backend_resnet_size=50, norm_type='instancenorm', dropout=0.0, use_blurpool=True, use_resnet_cd=True
     MSCNN1Datt_catt_2DResNetCBAM: frontend_filters=32, frontend_kernel=64, cross_attention_heads=4, backend_resnet_size=50, norm_type='instancenorm', use_blurpool=True, use_resnet_cd=True

.. code-block:: console

   $ sage list losses
   Losses (parameters with their defaults):
     BCE_PE_Sigma: regression_weight=0.005, coupling_weight=0.005, beta=0.5, sigma_min=0.001, sigma_max=10.0, confidence_weighting=True
     BCE_PE_Huber: regression_weight=1.0, confidence_weighting=True
     BCE_PE_MSE: regression_weight=1.0, confidence_weighting=False

.. code-block:: console

   $ sage list waveforms
   Waveforms (parameters with their defaults):
     IMRPhenomPv2: no parameters
     IMRPhenomXPHM: no parameters
     IMRPhenomXAS_NRTidalv3: no parameters

.. dropdown:: sage list optimisers

   .. code-block:: console

      $ sage list optimisers
      Optimisers (parameters with their defaults):
        Adafactor: lr=0.01, beta2_decay=-0.8, eps=(None, 0.001), d=1.0, weight_decay=0.0, foreach=None, maximize=False
        Adadelta: lr=1.0, rho=0.9, eps=1e-06, weight_decay=0, foreach=None, capturable=False, maximize=False, differentiable=False
        Adagrad: lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10, foreach=None, maximize=False, differentiable=False
        Adam: lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, decoupled_weight_decay=False
        Adamax: lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, foreach=None, maximize=False, differentiable=False, capturable=False
        AdamW: lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False, maximize=False, foreach=None, capturable=False, differentiable=False
        ASGD: lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0, foreach=None, maximize=False, differentiable=False, capturable=False
        LBFGS: lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None
        NAdam: lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004, decoupled_weight_decay=False, foreach=None, maximize=False, capturable=False, differentiable=False
        RAdam: lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, decoupled_weight_decay=False, foreach=None, maximize=False, capturable=False, differentiable=False
        RMSprop: lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, capturable=False, foreach=None, maximize=False, differentiable=False
        Rprop: lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50), capturable=False, foreach=None, maximize=False, differentiable=False
        SGD: lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False, maximize=False, foreach=None, differentiable=False
        SparseAdam: lr=0.001, betas=(0.9, 0.999), eps=1e-08, maximize=False

.. dropdown:: sage list schedules

   .. code-block:: console

      $ sage list schedules
      Schedules (parameters with their defaults):
        WarmupPolynomialDecayFlatTail: warmup_steps=20000, warmup_start_factor=0.001, decay_power=2.0, min_lr=1e-06, flat_tail_epochs=4
        LambdaLR: step_every (required: batch, epoch, metric), lr_lambda (required)
        MultiplicativeLR: step_every (required: batch, epoch, metric), lr_lambda (required)
        StepLR: step_every (required: batch, epoch, metric), step_size (required), gamma=0.1
        MultiStepLR: step_every (required: batch, epoch, metric), milestones (required), gamma=0.1
        ConstantLR: step_every (required: batch, epoch, metric), factor=0.3333333333333333, total_iters=5
        LinearLR: step_every (required: batch, epoch, metric), start_factor=0.3333333333333333, end_factor=1.0, total_iters=5
        ExponentialLR: step_every (required: batch, epoch, metric), gamma (required)
        SequentialLR: step_every (required: batch, epoch, metric), schedulers (required), milestones (required)
        PolynomialLR: step_every (required: batch, epoch, metric), total_iters=5, power=1.0
        CosineAnnealingLR: step_every (required: batch, epoch, metric), T_max (required), eta_min=0.0
        ChainedScheduler: step_every (required: batch, epoch, metric), schedulers (required)
        ReduceLROnPlateau: step_every (required: batch, epoch, metric), mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
        CyclicLR: step_every (required: batch, epoch, metric), base_lr (required), max_lr (required), step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9
        CosineAnnealingWarmRestarts: step_every (required: batch, epoch, metric), T_0 (required), T_mult=1, eta_min=0.0
        OneCycleLR: step_every (required: batch, epoch, metric), max_lr (required), total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, three_phase=False

.. dropdown:: sage list asds

   .. code-block:: console

      $ sage list asds
      Analytic ASDs (synthetic noise):
        AdVBNSOptimizedSensitivityP1200087
        AdVDesignSensitivityP1200087
        AdVEarlyHighSensitivityP1200087
        AdVEarlyLowSensitivityP1200087
        AdVLateHighSensitivityP1200087
        AdVLateLowSensitivityP1200087
        AdVMidHighSensitivityP1200087
        AdVMidLowSensitivityP1200087
        AdVO3LowT1800545
        AdVO4IntermediateT1800545
        AdVO4T1800545
        AdvVirgo
        CosmicExplorerP1600143
        CosmicExplorerPessimisticP1600143
        CosmicExplorerWidebandP1600143
        EinsteinTelescopeP1600143
        GEO
        GEOHF
        KAGRA
        KAGRA128MpcT1800545
        KAGRA25MpcT1800545
        KAGRA80MpcT1800545
        KAGRADesignSensitivityT1600593
        KAGRAEarlySensitivityT1600593
        KAGRALateSensitivityT1600593
        KAGRAMidSensitivityT1600593
        KAGRAOpeningSensitivityT1600593
        TAMA
        Virgo
        aLIGO140MpcT1800545
        aLIGO175MpcT1800545
        aLIGOAPlusDesignSensitivityT1800042
        aLIGOAdVO3LowT1800545
        aLIGOAdVO4IntermediateT1800545
        aLIGOAdVO4T1800545
        aLIGOBHBH20Deg
        aLIGOBHBH20DegGWINC
        aLIGOBNSOptimizedSensitivityP1200087
        aLIGODesignSensitivityP1200087
        aLIGODesignSensitivityT1800044
        aLIGOEarlyHighSensitivityP1200087
        aLIGOEarlyLowSensitivityP1200087
        aLIGOHighFrequency
        aLIGOHighFrequencyGWINC
        aLIGOKAGRA128MpcT1800545
        aLIGOKAGRA25MpcT1800545
        aLIGOKAGRA80MpcT1800545
        aLIGOLateHighSensitivityP1200087
        aLIGOLateLowSensitivityP1200087
        aLIGOMidHighSensitivityP1200087
        aLIGOMidLowSensitivityP1200087
        aLIGONSNSOpt
        aLIGONSNSOptGWINC
        aLIGONoSRMHighPower
        aLIGONoSRMLowPower
        aLIGONoSRMLowPowerGWINC
        aLIGOO3LowT1800545
        aLIGOQuantumBHBH20Deg
        aLIGOQuantumHighFrequency
        aLIGOQuantumNSNSOpt
        aLIGOQuantumNoSRMHighPower
        aLIGOQuantumNoSRMLowPower
        aLIGOQuantumZeroDetHighPower
        aLIGOQuantumZeroDetLowPower
        aLIGOThermal
        aLIGOZeroDetHighPower
        aLIGOZeroDetHighPowerGWINC
        aLIGOZeroDetLowPower
        aLIGOZeroDetLowPowerGWINC
        aLIGOaLIGO140MpcT1800545
        aLIGOaLIGO175MpcT1800545
        aLIGOaLIGODesignSensitivityT1800044
        aLIGOaLIGOO3LowT1800545
        analytical_psd_lisa_tdi_AE
        analytical_psd_lisa_tdi_AE_confusion
        analytical_psd_lisa_tdi_T
        analytical_psd_lisa_tdi_XYZ
        analytical_psd_taiji_tdi_AE
        analytical_psd_taiji_tdi_AE_confusion
        analytical_psd_taiji_tdi_T
        analytical_psd_taiji_tdi_XYZ
        analytical_psd_tianqin_tdi_AE
        analytical_psd_tianqin_tdi_AE_confusion
        analytical_psd_tianqin_tdi_T
        analytical_psd_tianqin_tdi_XYZ
        eLIGOModel
        eLIGOShot
        flat_unity
        iLIGOModel
        iLIGOSRD
        iLIGOSeismic
        iLIGOShot
        iLIGOThermal
        sh_transformed_psd_lisa_tdi_XYZ

A few settings have choices without a ``sage list``:

.. list-table::
   :header-rows: 1
   :widths: 40 60
   :class: wrap

   * - Setting
     - Choices
   * - ``signal.projection``
     - ``constant``
   * - ``signal.snr``, ``validation.snr``
     - ``halfnorm``
   * - ``noise.training``, ``noise.validation``
     - ``real``, ``synthetic``
   * - ``preprocess`` steps
     - ``whiten``, ``minmax_normalise``, ``multirate``

More options will be added in the future. To request one, open an issue on
`GitHub <https://github.com/nnarenraju/sage/issues>`_.
