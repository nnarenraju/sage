Glossary
=========

.. glossary::
   :sorted:

   ASD
      Amplitude Spectral Density. The square root of the :term:`PSD`, often used to
      characterise detector noise in units of strain per root Hz.

   BCE
      Binary Cross-Entropy. The classification loss used by Sage to distinguish signal
      from noise samples. Combined with heteroscedastic regression in
      :class:`~sage.architecture.custom_losses.BCEWithPEsigmaLoss`.

   BBH
      Binary Black Hole. A compact binary system consisting of two black holes. BBH
      mergers are the dominant source class in current LIGO/Virgo catalogues and the
      primary target of Sage's production runs.

   BNS
      Binary Neutron Star. A compact binary system consisting of two neutron stars.
      BNS mergers produce both gravitational waves and electromagnetic counterparts
      (kilonovae), as demonstrated by GW170817.

   CBAM
      Convolutional Block Attention Module. An attention mechanism applied after each
      residual block in Sage's backend ResNet. Applies sequential channel attention
      and spatial attention to focus the network on merger-relevant features.

   CBC
      Compact Binary Coalescence. The inspiral, merger, and ringdown of a compact
      binary system (BBH, BNS, or NSBH). CBC signals are the primary target of
      current matched-filter and ML GW search pipelines.

   Chirp mass
      The combination of component masses :math:`m_1` and :math:`m_2` that governs
      the leading-order frequency evolution of a CBC signal:
      :math:`\mathcal{M} = (m_1 m_2)^{3/5} / (m_1 + m_2)^{1/5}`.
      It is the best-measured mass parameter in GW observations.

   FAR
      False Alarm Rate. The rate at which noise triggers exceed a given ranking
      statistic threshold, typically expressed in units of events per month or per year.
      Detection claims are usually made at FAR < 1 per year.

   FD
      Frequency Domain. Sage generates waveforms and performs whitening in the
      frequency domain for efficiency, converting to the time domain only for
      multirate compression.

   Fiducial PSD
      A fixed representative PSD used for whitening during training. Sage pre-computes
      a fiducial PSD from the noise dataset and applies it consistently across all
      batches to avoid per-sample PSD estimation overhead.

   GWOSC
      Gravitational-Wave Open Science Center. The public repository of LIGO/Virgo/KAGRA
      strain data, segment information, and event catalogues. Sage's data primer
      downloads directly from GWOSC.

   Heteroscedastic regression
      A regression approach in which the model predicts both the mean and the
      variance of each output. Sage uses this to produce calibrated uncertainty
      estimates on parameters such as chirp mass and coalescence time.

   IMRPhenomD
      An aligned-spin, frequency-domain phenomenological waveform model for CBC signals.
      Sage provides a fully vectorised GPU implementation compatible with
      ``torch.compile``.

   IMRPhenomPv2
      An extension of IMRPhenomD to precessing-spin binaries via the PhenomPv2
      "twist-up" formalism. The default waveform model used in Sage production runs.

   IST
      Inverse Spectrum Truncation. A time-domain windowing technique applied to the
      inverse PSD before whitening, preventing spectral leakage from sharp PSD features
      such as spectral lines.

   MLGWSC
      Machine Learning Gravitational-Wave Search Challenge. A standardised injection
      study (Schäfer et al. 2023) providing common noise and signal datasets for
      direct comparison between GW search pipelines.

   Mismatch
      A measure of waveform similarity: :math:`1 - \langle h_1, h_2 \rangle`, where
      the angle bracket denotes the noise-weighted inner product maximised over time
      and phase. A mismatch of zero means identical waveforms.

   Multirate sampling
      A compression scheme that assigns different sample rates to different time
      intervals of a CBC signal, keeping high time resolution only near the merger
      where signal power is concentrated. Reduces the effective sequence length from
      ~24 576 to ~2 600 samples.

   NSBH
      Neutron Star – Black Hole. A compact binary system consisting of one neutron
      star and one black hole.

   OTF
      On-the-Fly. Sage generates every training batch from scratch at runtime —
      no pre-computed dataset is stored. Waveforms and noise windows are sampled
      and processed per batch to eliminate data-reuse biases.

   PSD
      Power Spectral Density. A frequency-domain characterisation of detector noise
      power per unit bandwidth, in units of strain² per Hz. Used for whitening and
      optimal SNR calculation.

   Recolouring
      A data augmentation technique in which a real noise segment is whitened with its
      original PSD and then re-coloured with a different PSD drawn from a library. This
      exposes the network to a wider range of noise spectral shapes during training.

   ROC curve
      Receiver Operating Characteristic curve. A plot of true alarm probability versus
      false alarm probability as the detection threshold is varied. Used to compare
      pipeline sensitivity across different methods.

   Sensitive distance
      The volume-averaged maximum distance at which a pipeline can detect signals from
      a given population at a fixed :term:`FAR`. A standard MLGWSC benchmark metric.

   SNR
      Signal-to-Noise Ratio. The ratio of signal power to noise power in the detector
      data, optimally computed as the noise-weighted inner product of the signal
      template with itself. Sage uses optimal SNR rescaling during training.

   TD
      Time Domain. Real-valued strain as a function of time. Sage applies multirate
      sampling in the time domain after converting whitened frequency-domain data via
      inverse FFT.

   Whitening
      Division of the frequency-domain strain by the :term:`ASD`, transforming
      coloured detector noise into approximately white (flat-spectrum) noise. Sage
      implements whitening via :class:`~sage.dsp.whiten.FiducialWhitening`.
