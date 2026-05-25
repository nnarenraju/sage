Developer Guide
================

.. warning::

   This guide is under construction. The content below outlines what will be available here
   once the guide is complete.

----

The Developer Guide covers everything needed to contribute to Sage — from setting up a
development environment to adding a new waveform model or frontend architecture.

----

What Will Be Here
------------------

Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~

- Cloning the repo and installing in editable mode (``pip install -e .``)
- Setting up pre-commit hooks for linting and type checking
- Running the test suite locally

Code Conventions
~~~~~~~~~~~~~~~~~

- Module layout and naming conventions
- Docstring style (Google-style, rendered via Napoleon)
- Type annotation expectations
- When to use ``torch.compile``-safe patterns vs. when a graph break is acceptable

Extending Sage
~~~~~~~~~~~~~~~

Step-by-step guides for the most common extension points:

- **Adding a new waveform model** — implementing the sampler interface so the new model
  plugs into the existing training loop without changes
- **Adding a new frontend** — the multirate and multibanding frontends as worked examples
- **Adding a new backend** — swapping out the ResNet for a different architecture
- **Adding a new loss function** — the ``BCEWithPEsigmaLoss`` as a reference implementation
- **Adding a new noise sampler** — extending ``MemmapNoiseSampler`` for a custom dataset

Testing
~~~~~~~~

- What the test suite covers and how to run individual test groups
- Writing regression tests for new DSP components
- Using the MLGWSC-1 injection set as an integration test

CI / CD Pipeline
~~~~~~~~~~~~~~~~~

- What runs on every pull request (lint, type check, unit tests)
- What runs on merge to main (integration tests, doc build)
- How to read failing CI output and diagnose common failures

Release Process
~~~~~~~~~~~~~~~~

- Versioning convention (``MAJOR.MINOR.PATCH``)
- Steps to cut a new release: changelog, tag, PyPI upload, RTD version publish
- Backport policy for bug fixes

Contributing Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

- How to open a good bug report
- How to propose a new feature (GitHub discussion before PR)
- PR review expectations and merge criteria
