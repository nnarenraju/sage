Installation
============

Sage is currently intended for local editable installs. A CUDA-capable GPU is strongly
recommended for on-the-fly waveform generation, training, and large injection studies.

**Requirements**: Python ≥ 3.9, PyTorch ≥ 2.1, CUDA (optional but recommended).

Option A — conda (recommended for GPU clusters)
------------------------------------------------

The `utils/environment.yml <https://github.com/nnarenraju/sage/blob/main/utils/environment.yml>`_
file defines the full conda environment (Python 3.11, LALSuite, PyCBC, GWpy, and JupyterLab).
`utils/create_env.sh <https://github.com/nnarenraju/sage/blob/main/utils/create_env.sh>`_
automates the three-step setup: conda packages → PyTorch CUDA wheel → editable Sage install.

.. code-block:: bash

    git clone https://github.com/nnarenraju/sage.git
    cd sage/utils
    bash create_env.sh
    conda activate sage

Option B — pip only
-------------------

.. code-block:: bash

    git clone https://github.com/nnarenraju/sage.git
    cd sage
    python -m pip install -r requirements.txt
    python -m pip install -e .

.. note::

   PyTorch installation can depend on your CUDA version. If needed, install the
   appropriate PyTorch build first using the command from
   `pytorch.org <https://pytorch.org/get-started/locally/>`_, then install the
   remaining requirements.

Verifying the installation
--------------------------

Run a quick smoke test to confirm configuration registration works:

.. code-block:: bash

    python -c "
    from sage.core.config import register_configs
    from sage.presets.data_configs import Default as data_cfg
    from sage.presets.configs import DefaultConfig as cfg
    register_configs(cfg, data_cfg)
    print('Config registration: OK')
    "

To run the full test suite:

.. code-block:: bash

    pytest tests/ -v

For a broad syntax check across the package:

.. code-block:: bash

    python -m py_compile $(find sage -name '*.py')
