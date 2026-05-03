#!/bin/bash

# Stable cache dir avoids /tmp cleanup killing Triton temp files mid-compile
export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torchinductor"
export TRITON_CACHE_DIR="${HOME}/.cache/triton"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

# Serialize Triton kernel compilation to prevent temp-file race conditions
# under max-autotune; safe to remove once compilation succeeds and is cached.
export TORCHINDUCTOR_COMPILE_THREADS=1

# Run Sage O3b hard-mining training
python3 -c "from train import run_sage; run_sage()"
