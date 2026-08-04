Noise Dataset Generation
=========================

This guide covers everything needed to turn the downloaded ``.bin`` files into the
noise data structures used during training: batch sampling, PSD estimation, smoothing,
and recolouring to augment the spectral diversity of the training set.

It also covers :doc:`synthetic noise generation <synthetic_noise>` — simulated Gaussian
noise coloured by an ASD of your choosing, for tests, controlled experiments, and any
situation where you need to know the exact spectrum of the data.

.. toctree::
   :maxdepth: 1

   sampling
   fiducial_psds
   recolouring
   synthetic_noise
