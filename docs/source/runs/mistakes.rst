Common Errors
=============

If something in ``run.yaml`` or ``prior.yaml`` is wrong, Sage stops with two
lines: ``error`` says what is wrong and where, and ``fix`` says what to do
and which command to run again. For example, with ``epochs`` misspelt in
``run.yaml``:

.. code-block:: console

   $ sage check
   error: train: unknown setting 'epoch'; did you mean 'epochs'?
   fix: edit run.yaml as above, then: sage check

Mistakes in run.yaml
--------------------

A value of the wrong type:

.. code-block:: console

   $ sage check
   error: train.epochs: expected int, got 'six'
   fix: edit run.yaml as above, then: sage check

A setting left out:

.. code-block:: console

   $ sage check
   error: train: missing setting 'batch_size'
   fix: edit run.yaml as above, then: sage check

An unknown name for a part of Sage:

.. code-block:: console

   $ sage check
   error: signal.waveform.name: unknown 'IMRPhenomPv3'; did you mean 'IMRPhenomPv2'?
   fix: edit run.yaml as above, then: sage check

A parameter the part does not have:

.. code-block:: console

   $ sage check
   error: model: unknown parameter 'kernel'; valid: frontend_filters, frontend_kernel, backend_resnet_size, norm_type, dropout, use_blurpool, use_resnet_cd
   fix: edit run.yaml as above, then: sage check

A network and a loss that do not fit together:

.. code-block:: console

   $ sage check
   error: model MSCNN1Datt_2DResNetCBAM gives an estimate and a width per target, but loss BCE_PE_Huber needs one estimate per target; use a loss from ['BCE_PE_Sigma'] or a model from ['MSCNN1D_2DResNetCBAM', 'MSCNN1Datt_catt_2DResNetCBAM']
   fix: edit run.yaml as above, then: sage check

A line indented wrongly. The message points at the line and column:

.. code-block:: console

   $ sage check
   error: while parsing a block mapping
     in "<unicode string>", line 4, column 2:
        detectors: [H1, L1]
        ^
   expected <block end>, but found '<block mapping start>'
     in "<unicode string>", line 5, column 3:
         sample_rate_in_hz: 2048.0
         ^
   fix: edit run.yaml as above, then: sage check

Mistakes in prior.yaml
----------------------

These name ``prior.yaml`` in both lines.

An unknown distribution:

.. code-block:: console

   $ sage check
   error: prior.yaml: unknown distribution 'flat' for prior 'delta_tc'; allowed: ['sin_angle', 'uniform', 'uniform_angle', 'uniform_comoving_volume', 'uniform_radius', 'uniform_sky', 'uniform_solidangle']
   fix: edit prior.yaml as above, then: sage check

A bound that is not a number:

.. code-block:: console

   $ sage check
   error: prior.yaml: prior 'mass1': min must be a number, got 'seven'
   fix: edit prior.yaml as above, then: sage check

A parameter drawn by the priors but left out of ``variable_params``:

.. code-block:: console

   $ sage check
   error: prior.yaml: variable_params does not match the parameters the priors draw: missing ['delta_tc'], not drawn []
   fix: edit prior.yaml as above, then: sage check

A missing ``prior.yaml``:

.. code-block:: console

   $ sage check
   error: no prior file prior.yaml
   fix: put the run's prior file back at prior.yaml, then: sage check

After training starts
---------------------

Once training has started, the settings are fixed, and a command that finds
them changed stops and points to ``sage clone``. :doc:`cloning` explains
why, and how to carry the changes into a new run.
