#!/bin/bash
# ===========================================================================
# runs/o3b/submit.sh -- thin per-run launcher.
#
# All the heavy lifting (paths, conda python, SLURM flags) lives in
# sage/utils/run_base.sh, driven by the server registry in
# sage/utils/servers.py. Here we only:
#   1. pick the server (one line), and
#   2. choose which task to launch.
#
# Run it from this directory:   ./submit.sh download
# ===========================================================================

# --- 1. pick the server -----------------------------------------------------
# Leave unset to auto-detect from the hostname; set explicitly to override.
# export SAGE_SERVER=jarvis

# --- load the shared launch library ----------------------------------------
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
source "$REPO_ROOT/sage/utils/run_base.sh"

# --- 2. choose the task -----------------------------------------------------
TASK="${1:-download}"   # download | retry | psds | train

# Data tasks are network/CPU-bound (GWOSC download + PSD estimation runs on
# CPU), so they go to the cpu partition with no GPU -- not the GPU node the
# registry defaults to for training.
CPU_OPTS=(--partition cpu --qos "" --gres none --cpus 16 --mem 64G --time 2-00:00)

case "$TASK" in
    download)
        # Full O3b download (H1, L1, V1) + PSD generation.
        sage_submit "${CPU_OPTS[@]}" --job o3b-download \
            "python -c 'from dataset import make_dataset; make_dataset()'"
        ;;
    retry)
        # Re-fetch any missing/failed segments (retry lives in the downloader).
        sage_submit "${CPU_OPTS[@]}" --job o3b-retry \
            "python -c 'from dataset import retry_dataset; retry_dataset(num_workers=8)'"
        ;;
    psds)
        # Regenerate PSDs only (assumes .bin files already downloaded).
        sage_submit "${CPU_OPTS[@]}" --job o3b-psds \
            "python -c 'from dataset import make_psds_only; make_psds_only()'"
        ;;
    train)
        sage_submit --job o3b-train \
            "python -c 'from train import run_sage; run_sage()'"
        ;;
    *)
        echo "Unknown task '$TASK'. Use: download | retry | psds | train" >&2
        exit 2
        ;;
esac
