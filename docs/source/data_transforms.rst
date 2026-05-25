Data Transforms
================

Once a combined signal+noise batch leaves the on-the-fly generators (see
:doc:`otf_data_generation`), it passes through a fixed preprocessing chain before
reaching the network. The chain is managed by :class:`~sage.core.graph.Preprocessor`,
which applies its transforms in sequence.

The two transforms are:

1. :class:`~sage.dsp.whiten.FiducialWhitening` — flatten the noise PSD.
2. :class:`~sage.dsp.multirate_sampling.MultirateSampler` — compress the
   time axis using dyadic multirate sampling.

Building the preprocessor
--------------------------

In a production run the preprocessing configuration is derived directly from the
parameter sampler's bounds, ensuring the multirate bins always cover the full
prior mass range:

.. code-block:: python

    from sage.dsp.whiten import FiducialWhitening
    from sage.dsp.multirate_sampling import MultirateSampler, DyadicPyramidBinning
    from sage.core.graph import Preprocessor

    # Derive multirate bins from the waveform prior bounds
    # param_sampler.bounds is a dict {name: (min, max)} — needs mass1/mass2 entries
    dyadic_binning = DyadicPyramidBinning(param_sampler.bounds)
    mrsampler = MultirateSampler(binning_method=dyadic_binning)

    # Whitener reads fiducial PSDs from cfg.export_dir automatically
    whitener = FiducialWhitening()

    # Preprocessor applies transforms left to right
    processor = Preprocessor([whitener, mrsampler])

The :class:`~sage.dsp.multirate_sampling.DyadicPyramidBinning` computes the
time–frequency bins analytically from the prior's mass bounds so that the
highest-rate bin always covers the full range of possible coalescence times for
every waveform in the prior.

Applying the preprocessor
--------------------------

.. code-block:: python

    # combined: (B, n_det, n_freq) — complex FD batch from otf_data_generation
    x = processor(combined)
    # x: (B, n_det, n_compressed_samples) — real TD, multirate-compressed

The output ``x`` is a real-valued, time-domain tensor with the merger region
preserved at the full sample rate and the early inspiral compressed. This is the
direct input to the network.

What each transform does
-------------------------

**FiducialWhitening**

Divides each frequency bin by ``sqrt(fiducial_PSD * df)``, applies an IFFT to
produce a time-domain output, and trims both ends to remove filter roll-on artefacts.
Uses the same fiducial PSD for every sample in the batch, which means it can run as a
single GPU-fused kernel — there is no per-sample PSD lookup.

The fiducial PSD is the median of 250 000 Welch PSDs estimated from the raw noise data
(see :doc:`noise_dataset/fiducial_psds`). Using the median rather than the mean makes
the whitener robust to glitch-contaminated segments that would otherwise inflate the
estimated noise floor.

**MultirateSampler**

Takes the whitened time-domain batch and applies the multirate bins computed by
``DyadicPyramidBinning``. Each bin covers a time interval of the segment and is
downsampled to a lower sample rate. The result is a concatenated 1D representation
where the time axis is short (~2600 samples) compared to the raw whitened segment
(~24 576 samples).

The bin boundaries shift with the prior mass range: the lighter the minimum mass, the
earlier the inspiral starts to carry signal power, and the longer the high-resolution
region must be. Passing ``param_sampler.bounds`` ensures the bins always reflect the
actual training prior.

Passing the preprocessor to the training loop
-----------------------------------------------

The ``processor`` object is passed directly to :class:`~sage.factory.training.SageUncompiledTraining`
and :class:`~sage.factory.validation.SageUncompiledValidation`. Inside the loop,
``processor(x)`` is called once per batch after signal injection:

.. code-block:: python

    # Inside SageUncompiledTraining.forward():
    x = noise_data + signal_pad     # (B, n_det, n_freq) — FD
    x = self.processor(x)           # (B, n_det, compressed_samples) — TD compressed
    out = self.model(x)
