Custom Loss Functions
======================

Sage provides two multi-task loss functions that combine binary classification of
detection with continuous parameter estimation.  Both are subclasses of
:class:`torch.nn.Module` and return a stacked tensor of loss components so the
training loop can track each term separately.

All losses expect the network to return a tuple ``(ranking_stat, point_estimates)``
and a target tensor of shape ``(B, num_pe + 1)`` where the last column is the
binary class label (``0`` = noise, ``1`` = signal).

.. toctree::
   :maxdepth: 1

   bce_reg
   bce_sigma
