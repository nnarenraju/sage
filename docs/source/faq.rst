Frequently Asked Questions
===========================

.. dropdown:: My GPU is on fire — should I be worried?

   Only if it was not already on fire before you started training. In all other cases,
   reduce your batch size, check your cooling, and consider whether
   ``torch.backends.cudnn.benchmark = True`` is appropriate for your hardware.
   If the fire persists, please prioritise contacting your local fire department over
   opening a GitHub issue.

.. dropdown:: My loss is converging but my collaborator insists the results are biased. Who is right?

   Both. A converging loss confirms the network is learning something — it does not confirm
   the network is learning the *right* thing. Sage addresses 11 interconnected
   supervised-learning biases documented in the paper; if you believe you have found a 12th,
   we genuinely welcome a GitHub issue.

.. dropdown:: My partner says I spend more time with Sage than with them. Should I be worried?

   Define "worried." If your validation loss is still decreasing, you are making measurable
   progress on at least one front. If you need to find someone who is
   a better match for you, we recommend the matched-filter pipelines for theoretical guarantees
   in the mismatch. There are plenty of templates in the bank. She's not the one for you.

----

.. note::

   More questions coming soon. If you have a question that keeps coming up,
   open a `GitHub discussion <https://github.com/nnarenraju/sage/issues>`__
   and it may end up here.
