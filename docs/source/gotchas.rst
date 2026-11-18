Gotchas
=======

Known issues
------------

.. dropdown:: Why does training diverge when a preprocessing step is trained with the network?

   When the network is compiled and its input carries gradients, training can diverge:
   the loss grows steadily from the start. A preprocessing step with trainable
   parameters passes gradients back through the network's input, so it triggers the
   issue, even if those parameters never change.

   It has been seen with ``train.compile`` set to ``mode: max-autotune``. Whether it comes
   from CUDA graphs or from the compiled kernels is not yet known.

   Until then, train such runs without compiling the network. Training is slower without it.

   .. code-block:: yaml

      train:
        compile: false
