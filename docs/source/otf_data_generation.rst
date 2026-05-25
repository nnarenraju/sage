On-the-Fly Data Generation Pipeline
======================================

Sage generates every training batch from scratch at runtime — no fixed dataset is
stored on disk. This page shows how to assemble the two data generators (signal and
noise) into a self-contained pipeline that produces ready-to-use frequency-domain
batches at each call.

The full pipeline for one batch is:

.. figure:: /_static/methodology-flowchart.png
   :align: center
   :alt: Sage methodology flowchart

   Flowchart of the Sage on-the-fly generation pipeline. Elements in the dotted box
   (configuration and data priming) are set once per run. The signal generation branch
   is specific to the signal class; common settings apply to both signal and noise.
   All boxes with solid outlines are iterated over at every training batch.

Signal sampler with SNR rescaling
-----------------------------------

The SNR rescaler draws a target SNR from a half-normal prior and rescales each waveform
amplitude to match it before injection. This controls the SNR distribution of the
training set independently of the distance prior.

.. code-block:: python

    from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2
    from sage.data.waveform import HalfNorm
    from sage.data.waveform.snr import OptimalSNRRescaler

    # Parameter prior from YAML
    param_sampler = read_from_config("./gwconfig.yaml", seed=150914)

    # Sky/polarisation projection (constant in time)
    waveform_project = ConstantProjection()

    # SNR prior: half-normal with loc=5, scale=4 → most injections SNR 5–20
    target_snr_sampler = HalfNorm(scale=4.0, loc=5.0, seed=150914)
    snrscaler = OptimalSNRRescaler(target_snr_sampler)

    # Combine into a signal sampler with automatic SNR rescaling
    signal_sampler = IMRPhenomPv2(
        param_sampler,
        waveform_project,
        augment=snrscaler,   # <— rescaling applied after waveform generation
    )

Calling ``signal_sampler()`` returns a ``(signal_data, signal_targets)`` pair:

.. code-block:: python

    signal_data, signal_targets = signal_sampler()
    # signal_data:    (S, n_detectors, n_freq)  — complex FD strain, rescaled
    # signal_targets: (S, num_pe + 1)           — regression targets + class label (1)

where ``S = batch_size * class_balance`` (default 50% of the batch).

Changing the SNR distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``HalfNorm(scale=s, loc=l)`` draws from a half-normal with lower bound ``l`` and
spread ``s``. To target louder injections (e.g. for early training):

.. code-block:: python

    target_snr_sampler = HalfNorm(scale=8.0, loc=8.0, seed=150914)

To inject at a single fixed SNR:

.. code-block:: python

    from sage.data.waveform.distributions.snr_rescaling import FixedSNRRescaler
    snrscaler = FixedSNRRescaler(target_snr=10.0)

Noise sampler with recolouring
--------------------------------

The noise sampler loads batches from the pre-downloaded ``.bin`` files and optionally
recolours them to a PSD drawn from the recolouring library.

.. code-block:: python

    from sage.data.noise import MemmapNoiseSampler, RecolourPostprocess

    recolour = RecolourPostprocess(
        p_recolour=0.37,                              # 37% of batches are recoloured
        recolour_dataset_dir="/path/to/data_dir",     # contains recolour_psds/ and fiducial_psds/
    )

    noise_sampler = MemmapNoiseSampler(
        postprocess_fn=recolour,
        prefetch=8,
        seed=150914,
    )

``noise_sampler()`` returns a ``(noise_data, noise_targets)`` pair:

.. code-block:: python

    noise_data, noise_targets = noise_sampler()
    # noise_data:    (B, n_detectors, n_freq)  — complex FD noise (whitened, FD)
    # noise_targets: (B, 1)                   — class label (0 = noise)

Disabling recolouring (e.g. for validation):

.. code-block:: python

    noise_sampler_val = MemmapNoiseSampler(
        postprocess_fn=None,   # no recolouring — use fiducial whitening only
        prefetch=8,
        seed=170817,
    )

Assembling a full batch manually
----------------------------------

The training loop injects signals into random noise positions:

.. code-block:: python

    import torch

    cfg = get_cfg()
    B = cfg.batch_size                          # total batch size
    S = int(B * cfg.class_balance)             # number of signal slots (default 50%)

    signal_data, signal_targets = signal_sampler()
    noise_data,  noise_targets  = noise_sampler()

    # Random injection positions
    idx = torch.randperm(B, device=cfg.device)[:S]

    signal_pad = torch.zeros_like(noise_data)
    signal_pad[idx] = signal_data              # inject into selected noise slots

    combined = noise_data + signal_pad         # (B, n_det, n_freq) — FD strain

The ``combined`` tensor is now a mixed batch: ``S`` slots contain signal+noise, the
remaining ``B-S`` slots contain pure noise. This is the input to the preprocessing
pipeline described in :doc:`data_transforms`.

Controlling randomness and reproducibility
-------------------------------------------

Every random process in the pipeline is seeded:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Component
     - Seed argument
   * - ``read_from_config(path, seed=...)``
     - Seeds the :class:`torch.Generator` inside the parameter sampler.
   * - ``HalfNorm(scale=..., loc=..., seed=...)``
     - Seeds the SNR prior sampler.
   * - ``MemmapNoiseSampler(seed=...)``
     - Seeds the noise index selection.

Use different seeds for training and validation to ensure the two streams never
overlap. The O3b run uses ``seed=150914`` for training and ``seed=170817`` for
validation throughout.
