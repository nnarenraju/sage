Signal Dataset Generation
==========================

Sage provides two GPU-native, fully vectorised waveform generators. Both run inside
``torch.compile`` and produce batches of frequency-domain strain directly on the GPU,
eliminating any CPU bottleneck during training.

.. toctree::
   :maxdepth: 1

   imrphenomd
   imrphenompv2
