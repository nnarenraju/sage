#!/bin/bash
# ===========================================================================
# runs/o4b/submit.sh -- thin per-run launcher (see o3b/submit.sh for the model).
# Heavy lifting (paths, conda python, SLURM flags) lives in
# sage/utils/run_base.sh, driven by sage/utils/servers.py.
#   ./submit.sh download    # or: retry | psds | train
# ===========================================================================

# export SAGE_SERVER=jarvis     # unset = auto-detect from hostname

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
source "$REPO_ROOT/sage/utils/run_base.sh"

TASK="${1:-download}"   # download | retry | psds | train

# Data tasks are network/CPU-bound -> cpu partition, no GPU.
CPU_OPTS=(--partition cpu --qos "" --gres none --cpus 16 --mem 64G --time 2-00:00)

case "$TASK" in
    download)
        sage_submit "${CPU_OPTS[@]}" --job o4b-download \
            "python -c 'from dataset import make_dataset; make_dataset()'"
        ;;
    retry)
        sage_submit "${CPU_OPTS[@]}" --job o4b-retry \
            "python -c 'from dataset import retry_dataset; retry_dataset(num_workers=8)'"
        ;;
    psds)
        sage_submit "${CPU_OPTS[@]}" --job o4b-psds \
            "python -c 'from dataset import make_psds_only; make_psds_only()'"
        ;;
    train)
        sage_submit --job o4b-train \
            "python -c 'from train import run_sage; run_sage()'"
        ;;
    train_hard)
        # ONE hard-mining training segment (<=2-day wall, no chaining). Use for
        # a short run or a smoke test; use `chain` for the full production run.
        sage_submit --time 2-00:00 --job o4b-hard \
            "python -c 'from train_hard import run_hard; run_hard()'"
        ;;
    chain)
        # Full hard-mining run as N back-to-back <=2-day segments (default 4).
        # Each segment resumes from the latest checkpoint; trailing segments
        # no-op once training is done. Override N:  ./submit.sh chain 5
        N="${2:-4}"
        sage_submit_chain "$N" --time 2-00:00 --job o4b-hard \
            "python -c 'from train_hard import run_hard; run_hard()'"
        ;;
    *)
        echo "Unknown task '$TASK'. Use: download | retry | psds | train | train_hard | chain [N]" >&2
        exit 2
        ;;
esac
