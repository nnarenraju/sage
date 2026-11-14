Changing a Run
==============

A run changes by editing ``run.yaml`` and ``prior.yaml`` before training
starts. Each example here starts from the ``run.yaml`` that ``sage init``
writes for ``quick``, and shows the whole section after the change, with the
changed lines highlighted.

When you change a part's ``name``, give the new part's own parameters:
``sage list`` shows them, and any you leave out take their defaults. Changes
to the prior are on :doc:`prior_file`.

Signals
-------

Another waveform
~~~~~~~~~~~~~~~~

To simulate the signals with IMRPhenomXPHM instead of IMRPhenomPv2:

.. code-block:: yaml
   :emphasize-lines: 5

   signal:
     low_frequency_cutoff_in_hz: 20.0
     reference_frequency_in_hz: 20.0
     waveform:
       name: IMRPhenomXPHM
     projection:
       name: constant
     snr:
       name: halfnorm
       scale: 4.0
       loc: 5.0
     targets: [delta_tc, mchirp]

More estimated parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

To have the network also estimate the mass ratio ``q``, add it to
``targets``. Every target is a parameter of the prior, and ``mchirp`` must
stay:

.. code-block:: yaml
   :emphasize-lines: 12

   signal:
     low_frequency_cutoff_in_hz: 20.0
     reference_frequency_in_hz: 20.0
     waveform:
       name: IMRPhenomPv2
     projection:
       name: constant
     snr:
       name: halfnorm
       scale: 4.0
       loc: 5.0
     targets: [delta_tc, mchirp, q]

The non-astrophysical class
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To train the network to tell non-astrophysical noise apart, as a third class
beside signal and noise, add ``non_astrophysical``:

.. code-block:: yaml
   :emphasize-lines: 13,14,15,16

   signal:
     low_frequency_cutoff_in_hz: 20.0
     reference_frequency_in_hz: 20.0
     waveform:
       name: IMRPhenomPv2
     projection:
       name: constant
     snr:
       name: halfnorm
       scale: 4.0
       loc: 5.0
     targets: [delta_tc, mchirp]
     non_astrophysical:
       noise_share: 0.2
       signal_noise_share: 0.8
       in_band_share: 0.75

Detectors and noise
-------------------

Another detector
~~~~~~~~~~~~~~~~

To add Virgo:

.. code-block:: yaml
   :emphasize-lines: 2

   data:
     detectors: [H1, L1, V1]
     sample_rate_in_hz: 2048.0
     sample_length_in_s: 12.0
     padding_length_in_s: 2.0
     dtype: float32

``sage download`` and ``sage build-asds`` then fetch and build Virgo's data
too.

Synthetic noise
~~~~~~~~~~~~~~~

To train and validate on simulated noise from a design sensitivity, instead
of real noise; ``sage list asds`` names the sensitivities:

.. code-block:: yaml
   :emphasize-lines: 3,4,5,6,7,8,10,11,12,13,14,15

   noise:
     training:
       name: synthetic
       asd:
         H1: aLIGODesignSensitivityP1200087
         L1: aLIGODesignSensitivityP1200087
       low_frequency_cutoff_in_hz: 15.0
       filter_duration_in_s: 128.0
     validation:
       name: synthetic
       asd:
         H1: aLIGODesignSensitivityP1200087
         L1: aLIGODesignSensitivityP1200087
       low_frequency_cutoff_in_hz: 15.0
       filter_duration_in_s: 128.0
     fiducial: [o3a, o3b]
     prefetch_batches: 8
     read_workers: 16

Recolouring
~~~~~~~~~~~

To recolour the training noise, add ``recolour``:

.. code-block:: yaml
   :emphasize-lines: 8,9,10,11,12

   noise:
     training:
       name: real
       runs: [o3b]
     validation:
       name: real
       runs: [o3a]
     recolour:
       target: o3a
       probability: 0.37
       asd_jitter_gain: 0.5
       validation: false
     fiducial: [o3a, o3b]
     prefetch_batches: 8
     read_workers: 16

Network and loss
----------------

A smaller network
~~~~~~~~~~~~~~~~~

To use a smaller network, lower ``backend_resnet_size`` from 50 to 18:

.. code-block:: yaml
   :emphasize-lines: 5

   model:
     name: MSCNN1Datt_2DResNetCBAM
     frontend_filters: 32
     frontend_kernel: 64
     backend_resnet_size: 18
     norm_type: instancenorm
     dropout: 0.0
     use_blurpool: true
     use_resnet_cd: true

Another loss
~~~~~~~~~~~~

``BCE_PE_Huber`` needs a network that gives one estimate per parameter, such
as ``MSCNN1D_2DResNetCBAM``, so both change:

.. code-block:: yaml
   :emphasize-lines: 2,12,13

   model:
     name: MSCNN1D_2DResNetCBAM
     frontend_filters: 32
     frontend_kernel: 64
     backend_resnet_size: 50
     norm_type: instancenorm
     dropout: 0.0
     use_blurpool: true
     use_resnet_cd: true

   loss:
     name: BCE_PE_Huber
     regression_weight: 1.0
     confidence_weighting: true

Training
--------

Longer training
~~~~~~~~~~~~~~~

To train for 20 epochs of 64,000 samples:

.. code-block:: yaml
   :emphasize-lines: 2,3

   train:
     epochs: 20
     samples_per_epoch: 64000
     batch_size: 64
     signal_fraction: 0.5
     device: cuda
     precision: bf16
     clip_norm: 100.0
     compile: false
     seed: 150914
     keep_checkpoints: 2

A smaller learning rate
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml
   :emphasize-lines: 3

   optimiser:
     name: AdamW
     lr: 0.0001
     betas: [0.9, 0.999]
     eps: 1.0e-08
     weight_decay: 0.0001
     amsgrad: false
     maximize: false
     foreach: null
     capturable: false
     differentiable: false

A learning-rate schedule
~~~~~~~~~~~~~~~~~~~~~~~~

To lower the learning rate along a cosine over the 6 epochs, add a
``schedule`` section before ``train``. A PyTorch schedule also takes
``step_every``: ``batch``, ``epoch`` or ``metric``.

.. code-block:: yaml

   schedule:
     name: CosineAnnealingLR
     step_every: epoch
     T_max: 6
     eta_min: 0.0

Mining and averaging
--------------------

Hard-noise mining
~~~~~~~~~~~~~~~~~

To turn on hard-noise mining, add a ``mining`` section. Settings you leave
out take the defaults shown:

.. code-block:: yaml

   mining:
     enabled: true
     start_epoch: 3
     every_n_epochs: 4
     iterations: 8000
     keep_threshold: 1.0
     replay:
       probability: 0.2
       replays_per_epoch: 2.0
       focal_gamma: 3.0
       start_jitter_in_s: 2.0
     archive:
       n_cells: 2048
       descriptor_dim: 8
       novelty_dist: 0.15
       novelty_weight: 1.5
       max_embeddings: 50000
     emitter:
       n_emitters: 1
       batch_size: 72
       learning_rate: 0.1
       warmup: 2048

Weight averaging
~~~~~~~~~~~~~~~~

To average the network's weights over training, add an ``ema`` section:

.. code-block:: yaml

   ema:
     enabled: true
     decay: 0.9999
     calibration_batches: 500

Validation
----------

To validate after every epoch instead of every second one:

.. code-block:: yaml
   :emphasize-lines: 3

   validation:
     samples: 6400
     every_n_epochs: 1
     final_samples: 25600
     seed: 170817
     snr:
       name: halfnorm
       scale: 4.0
       loc: 5.0

Sage checks every change when it next reads the run, and stops with a message
if something is wrong; :doc:`mistakes` shows these messages. Every setting is
listed in :doc:`../configuration`.
