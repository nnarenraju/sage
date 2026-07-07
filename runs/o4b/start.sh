#!/bin/bash
# ===========================================================================
# runs/o3b/start.sh -- run a task NOW, in the foreground (no scheduler).
#
# Use this on an interactive/GPU node. It activates the server's conda env and
# paths via the shared library, then runs the chosen task in this shell.
# For batch submission use ./submit.sh instead.
#
#   ./start.sh download    # or: retry | psds | train (default)
# ===========================================================================

# export SAGE_SERVER=potsdam     # unset = auto-detect from hostname

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
source "$REPO_ROOT/sage/utils/run_base.sh"

TASK="${1:-train}"   # download | retry | psds | train

case "$TASK" in
    download) sage_run "python -c 'from dataset import make_dataset; make_dataset()'" ;;
    retry)    sage_run "python -c 'from dataset import retry_dataset; retry_dataset(num_workers=8)'" ;;
    psds)     sage_run "python -c 'from dataset import make_psds_only; make_psds_only()'" ;;
    train)      sage_run "python -c 'from train import run_sage; run_sage()'" ;;
    train_hard) sage_run "python -c 'from train_hard import run_hard; run_hard()'" ;;
    *) echo "Unknown task '$TASK'. Use: download | retry | psds | train | train_hard" >&2; exit 2 ;;
esac
