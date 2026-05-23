#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# create_env.sh — create the "sage" conda environment from scratch
#
# Usage:
#   bash create_env.sh
#
# What it does:
#   1. Creates the conda env from environment.yml  (conda-forge packages)
#   2. pip-installs PyTorch with CUDA 11.8 wheels  (not on conda-forge)
#   3. pip-installs sage in editable mode
#   4. Registers the env as a Jupyter kernel
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)   # utils/
REPO=$(cd "${SCRIPT_DIR}/.." && pwd)        # repo root (one level up)

# Auto-detect conda; fall back to explicit path if not on PATH
if command -v conda &>/dev/null; then
    CONDA=conda
else
    CONDA=/home/neuweiler/miniforge3/bin/conda
fi
ENV_NAME=sage
ENV_DIR="${HOME}/.conda/envs/${ENV_NAME}"

echo "=== Sage environment setup ==="
echo "  Repo     : ${REPO}"
echo "  Env dir  : ${ENV_DIR}"
echo "  Conda    : $($CONDA --version)"
echo

# ---------------------------------------------------------------------------
# Step 1 — create / update the conda environment
# ---------------------------------------------------------------------------
if $CONDA env list | grep -q "^${ENV_NAME} "; then
    echo "[1/4] Updating existing env '${ENV_NAME}' ..."
    $CONDA env update --name "${ENV_NAME}" --file "${SCRIPT_DIR}/environment.yml" --prune
else
    echo "[1/4] Creating env '${ENV_NAME}' in ${ENV_DIR} ..."
    $CONDA env create --name "${ENV_NAME}" --file "${SCRIPT_DIR}/environment.yml"
fi

PIP="${ENV_DIR}/bin/pip"
PYTHON="${ENV_DIR}/bin/python"

# ---------------------------------------------------------------------------
# Step 2 — PyTorch (CUDA 11.8)
#   conda-forge only carries cu120+ builds; the cu118 wheel must come from
#   the official PyTorch index, matching the cu118 build in the niamh env.
# ---------------------------------------------------------------------------
echo
echo "[2/4] Installing PyTorch (cu118) via pip ..."
"${PIP}" install \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu118 \
    --upgrade

# ---------------------------------------------------------------------------
# Step 3 — Sage (editable install)
# ---------------------------------------------------------------------------
echo
echo "[3/4] Installing sage in editable mode ..."
"${PIP}" install -e "${REPO}"

# ---------------------------------------------------------------------------
# Step 4 — Register as a Jupyter kernel so notebooks can pick it up
# ---------------------------------------------------------------------------
echo
echo "[4/4] Registering Jupyter kernel '${ENV_NAME}' ..."
"${PYTHON}" -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (sage)"

echo
echo "=== Done ==="
echo "  Activate : conda activate ${ENV_NAME}"
echo "  Jupyter  : jupyter lab  (kernel → 'Python (sage)')"
echo
echo "  Smoke test:"
echo "    ${PYTHON} -c \"import lalsimulation, pycbc, torch, sage; print('all imports OK')\""
