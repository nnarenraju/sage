#!/bin/bash
# ===========================================================================
# runs/o3a/submit.sh -- thin per-run launcher.
#
# Heavy lifting (paths, conda python, SLURM flags) lives in
# sage/utils/run_base.sh, driven by sage/utils/servers.py. Here we only pick
# the server and the task.
#
# Run from this directory:   ./submit.sh download
# ===========================================================================

# --- 1. pick the server (unset = auto-detect from hostname) -----------------
# export SAGE_SERVER=jarvis

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
source "$REPO_ROOT/sage/utils/run_base.sh"

# --- 2. choose the task -----------------------------------------------------
TASK="${1:-download}"   # download | retry | psds | train

# Data tasks are network/CPU-bound -> cpu partition, no GPU (see o3b/submit.sh).
CPU_OPTS=(--partition cpu --qos "" --gres none --cpus 16 --mem 64G --time 2-00:00)

case "$TASK" in
    download)
        sage_submit "${CPU_OPTS[@]}" --job o3a-download \
            "python -c 'from dataset import make_dataset; make_dataset()'"
        ;;
    retry)
        sage_submit "${CPU_OPTS[@]}" --job o3a-retry \
            "python -c 'from dataset import retry_dataset; retry_dataset(num_workers=8)'"
        ;;
    psds)
        sage_submit "${CPU_OPTS[@]}" --job o3a-psds \
            "python -c 'from dataset import make_psds_only; make_psds_only()'"
        ;;
    train)
        # Vanilla trainer. Optional 2nd arg: config module (default config).
        CFG="${2:-config}"
        sage_submit --job "o3a-$CFG-vanilla" \
            "SAGE_CONFIG='$CFG' python -c 'from train import run_sage; run_sage()'"
        ;;
    train_hard)
        # ONE hard-mining segment. Optional 2nd arg: config module to train
        # (default config). e.g.  ./submit.sh train_hard config_LV
        CFG="${2:-config}"
        sage_submit --time 2-00:00 --job "o3a-$CFG" \
            "SAGE_CONFIG='$CFG' python -c 'from train_hard import run_hard; run_hard()'"
        ;;
    chain)
        # Full run as chained <=2-day segments.  ./submit.sh chain [config] [N]
        # e.g.  ./submit.sh chain config_LV        (default config, N=4)
        CFG="${2:-config}"; N="${3:-4}"
        sage_submit_chain "$N" --time 2-00:00 --job "o3a-$CFG" \
            "SAGE_CONFIG='$CFG' python -c 'from train_hard import run_hard; run_hard()'"
        ;;
    *)
        echo "Unknown task '$TASK'. Use: download | retry | psds | train | train_hard [config] | chain [config] [N]" >&2
        exit 2
        ;;
esac
