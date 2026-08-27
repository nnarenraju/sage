#!/bin/bash
# ===========================================================================
# runs/search/submit.sh -- launcher for a search campaign.
#
# Paths, conda python and SLURM flags come from sage/utils/run_base.sh, driven
# by the server registry in sage/utils/servers.py.
#
# The usual case is one call:
#     ./submit.sh search config_o4a_HL
#
# which runs everything from the trained network to the campaign figures.
# Individual stages are available for staged or repeat running, and per-event
# characterization is a separate task because it needs its own environment.
# ===========================================================================

# --- server (leave unset to auto-detect from the hostname) ------------------
# export SAGE_SERVER=jarvis

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
source "$REPO_ROOT/sage/utils/run_base.sh"

TASK="${1:-help}"
# config_o3a_HL is the live campaign: an O3b-trained network searching O3a. config_o4a_HL
# exists with the same shape but raises until a network is trained on O4a.
CONFIG="${2:-config_o3a_HL}"

# Keep scratch off the system temporary directory: pytest, tarfile, matplotlib
# and astropy all default there, and it is not sized for this.
SCRATCH="/work/nagarajan/sage_runs/search/.scratch/$CONFIG"
export TMPDIR="$SCRATCH/tmp"
export MPLCONFIGDIR="$SCRATCH/mpl"
export XDG_CACHE_HOME="$SCRATCH/cache"
export ASTROPY_CACHE_DIR="$SCRATCH/astropy"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$ASTROPY_CACHE_DIR"

CPU_OPTS=(--partition cpu --qos "" --gres none --cpus 16 --mem 64G --time 2-00:00)
GPU_OPTS=(--cpus 16 --mem 128G --time 2-00:00)

# run_stage.py, not run_search.py: the two drivers do different jobs. run_search.py runs
# a whole track and takes no --stage; run_stage.py runs one named stage, and is what a
# staged or repeated submission wants.
# Set SAGE_AFTER to a job id to hold a submission until that job succeeds. Background
# follows zerolag -- the keep threshold is frozen from the *complete* zero-lag histogram
# before any slide is scored -- so the two are queued together and SLURM orders them,
# rather than the second being submitted by hand once the first is watched to the end.
after_opts () {
    [ -n "${SAGE_AFTER:-}" ] && printf -- "--dependency afterok:%s" "$SAGE_AFTER"
}

# SAGE_FORCE=1 re-runs a stage the manifest already records complete. Needed after a code
# change or a discarded product: the manifest is the record of what ran and deliberately
# not an inventory of the directory, so a deleted shard still reports complete and the
# stage exits in seconds having done nothing.
stage () {  # stage <name> <opts...>
    local name="$1"; shift
    local force=""
    [ -n "${SAGE_FORCE:-}" ] && force=" --force"
    sage_submit "$@" $(after_opts) --job "search-${name}-${CONFIG}" \
        "python -u run_stage.py --config $CONFIG --stage $name$force"
}

case "$TASK" in
    # --- the usual entry point ---------------------------------------------
    search)
        # Everything: score the run, background, rates, injections, sensitivity,
        # probabilities, candidates, catalogue comparison, figures and tables.
        sage_submit "${GPU_OPTS[@]}" --job "search-$CONFIG" \
            "python -u run_search.py --config $CONFIG"
        ;;
    smoke)
        # Same sequence with a shallow background, to exercise every step first.
        sage_submit "${GPU_OPTS[@]}" --time 12:00 --job "search-smoke-$CONFIG" \
            "python -u run_search.py --config $CONFIG --n-slides 8"
        ;;
    # --- every stage end to end at a shallow depth -------------------------
    shakedown)
        # Answers "does each stage run, and does what it writes satisfy the next".
        # Stops at catalogue: everything after it is still a stub.
        sage_submit "${GPU_OPTS[@]}" --time 12:00:00 \
            --job "search-shakedown-${CONFIG}" \
            "python -u run_search.py --config $CONFIG --stop-after catalogue"
        ;;

    plan)
        # Steps that would run and the projected cost. No submission.
        python run_search.py --config "$CONFIG" --dry-run
        ;;

    # --- individual stages, for staged or repeat running -------------------
    segments|grid|slides|far|sensitivity|pastro|trials|candidates|catalogue|store|figdata|figures|tables)
        stage "$TASK" "${CPU_OPTS[@]}"
        ;;
    zerolag|background|injections)
        stage "$TASK" "${GPU_OPTS[@]}"
        ;;
    # --- the background campaign, laid out across several GPUs -------------
    background-array)
        #     ./submit.sh background-array config_o3a_HL <n_slides> [gpus]
        #
        # One array task per contiguous group of slides, with `%GPUS` capping how
        # many run at once, so the campaign uses exactly the cards asked for and
        # no more. Slides within a task share one frontend cache build; slides in
        # different tasks do not, which is why the group is the unit and not the
        # slide.
        #
        # Requires, in order: zerolag complete (the keep threshold is frozen from
        # its histogram before any slide is scored) and, for the cache to be used
        # at all, `engine.use_frontend_cache` on with separability proven.
        SLIDES="${3:-82}"; GPUS="${4:-6}"; FIRST="${5:-1}"
        PER_TASK=$(( (SLIDES + GPUS - 1) / GPUS ))
        echo "background: $SLIDES slides over $GPUS GPUs, $PER_TASK slides per task"
        sage_submit "${GPU_OPTS[@]}" --array "1-${GPUS}%${GPUS}" $(after_opts) \
            --job "search-bg-${CONFIG}" \
            "python -u run_stage.py --config $CONFIG --stage background \
                 --slide-group \$SLURM_ARRAY_TASK_ID --slides-per-group $PER_TASK"
        ;;

    separability)
        # Proves the frontend cache is valid for this trained network, on the device the
        # campaign runs on. Worth 2.7x on the background, so it is measured on the
        # weights rather than read off the checkpoint's norm_type.
        sage_submit "${GPU_OPTS[@]}" --time 00:30:00 \
            --job "search-sep-${CONFIG}" \
            "python -m sage.diagnostics.diagnose_separability --config $CONFIG"
        ;;

    # --- how to lay a campaign out: measured, not assumed -------------------
    throughput-concurrency)
        # Does one worker saturate a card? If the aggregate rate is flat, the
        # right shape is one slide per GPU; if it scales, the device is waiting
        # on the host and the same cards carry several slides each.
        sage_submit "${GPU_OPTS[@]}" --time 01:00:00 \
            --job "search-conc-${CONFIG}" \
            "python -m sage.diagnostics.diagnose_search_throughput \
                 --config $CONFIG --windows ${3:-100000} --streams 1 2 3 4"
        ;;

    chain)
        # Background is the long pole; chain dependent jobs so it survives the
        # per-job wall-clock limit.
        sage_submit_chain "${3:-4}" "${GPU_OPTS[@]}" --job "search-bg-$CONFIG" \
            "python -u run_stage.py --config $CONFIG --stage background"
        ;;

    # --- per-event work, run afterwards ------------------------------------
    characterize)
        shift 2
        sage_submit "${CPU_OPTS[@]}" --cpus 32 --job "char-$CONFIG" \
            "python characterize.py --campaign $CONFIG $*"
        ;;

    # --- search-grade strain, built once per observing run -----------------
    dataprep)
        # Natural GWOSC segments, no chunking and no overlap.
        #     ./submit.sh dataprep O3a "H1 L1 V1" [chain]
        #
        # Submitted as a dependency chain. The build resumes from its own index,
        # so a segment that ends for any reason -- a wall limit, or a GWOSC
        # outage outlasting the in-process wait -- is continued by the next
        # without anyone watching. Once the release is complete the trailing
        # segments find nothing to do and exit in seconds.
        RUN="${2:-O3a}"; DETS="${3:-H1 L1 V1}"; CHAIN="${4:-4}"
        # 8 fetch workers: measured on a compute node, aggregate throughput
        # peaks there (21 MB/s) and falls at 4 (11) and at 16 (8, with a third
        # of transfers abandoned as too slow). 20 GB holds every O3a segment on
        # the single-pass conditioning path; --mem leaves room above it.
        # QOS "long" allows 14 days and no GPU, which is what this is; the
        # default "normal" caps at 2 days and would wall-kill a slow fetch.
        # Staging is inside the same quota: 24 files (48 with in-flight) is ~5.8 GiB
        # rather than ~15.5 GiB at 64, which is what the margin could not carry.
        # 32 GB holds the longest segment of every run (O4a L1 runs 74 h, which
        # needs 26 GB) on the single-pass conditioning path, so nothing falls
        # back to blocked conditioning. The nodes carry 773 GB.
        sage_submit_chain "$CHAIN" \
            --partition cpu --qos long --gres none --cpus 16 --mem 64G \
            --time 3-00:00 --job "dataprep-${RUN}" \
            "python -m sage.search.dataprep --run $RUN --detectors $DETS \
                 --flag DATA --memory-budget-gb 32 --workers 8 --cache-files 24 \
                 --outage-budget-s 7200"
        ;;
    dataprep-test)
        # A small search-grade release for exercising the pipeline, written to
        # HOME rather than project storage.
        #     ./submit.sh dataprep-test [run] [dets] [lo] [hi] [outdir]
        #
        # Same shape as the full release in every respect that the pipeline can
        # see -- natural GWOSC segments, no chunking, no overlap, all three
        # detectors, identical sidecar and HDF5 layout -- at ~9.5 GB instead of
        # 288 GB. The default span is chosen to contain four published O3a events
        # (GW190408_181802, GW190412, GW190413_052954, GW190413_134308), so
        # recovery is testable and not just the plumbing.
        RUN="${2:-O3a}"; DETS="${3:-H1 L1 V1}"
        LO="${4:-1238760000}"; HI="${5:-1239295000}"
        OUT="${6:-/home/nagarajan/o3a_test_search_data_DATA_HLV}"
        # Its own staging area: a paused full build owns the other one, and this
        # must not adopt or evict its staged files.
        sage_submit --partition cpu --qos long --gres none --cpus 8 --mem 48G \
            --time 1-00:00 --job "dataprep-test-${RUN}" \
            "python -m sage.search.dataprep --run $RUN --detectors $DETS --flag DATA \
                 --gps-range $LO $HI --out-dir '$OUT' \
                 --scratch-dir /work/nagarajan/sage_scratch/dataprep_test \
                 --memory-budget-gb 32 --workers 4 --cache-files 16 \
                 --outage-budget-s 7200"
        ;;
    dataprep-budget)
        # What it will cost, before fetching anything.
        python -m sage.search.dataprep --run "${2:-O3a}" \
            --detectors ${3:-H1 L1 V1} --flag DATA --budget
        ;;
    dataprep-verify)
        python -m sage.search.dataprep --run "${2:-O3a}" \
            --detectors ${3:-H1 L1 V1} --flag DATA --verify --checksums
        ;;

    # --- one-off measurement that sets the whole compute budget ------------
    throughput)
        # Measures f_full, f_front and f_back on real strain, and projects the
        # background cost with and without the frontend cache. Needs a GPU, and is
        # also the first time the engine touches the release -- so a failure here is
        # the zerolag smoke test failing, which is what it is for.
        WINDOWS="${3:-200000}"
        sage_submit "${GPU_OPTS[@]}" --time 01:00:00 \
            --job "search-throughput-${CONFIG}" \
            "python -m sage.diagnostics.diagnose_search_throughput \
                 --config $CONFIG --windows $WINDOWS"
        ;;

    throughput-sweep)
        # Background is the expensive stage and these cards hold 80-140 GB, so the
        # rate at the batch size the campaign will use is the number that matters.
        WINDOWS="${3:-400000}"
        sage_submit "${GPU_OPTS[@]}" --time 02:00:00 \
            --job "search-sweep-${CONFIG}" \
            "python -m sage.diagnostics.diagnose_search_throughput \
                 --config $CONFIG --windows $WINDOWS \
                 --sweep ${SWEEP:-1024 2048 4096 8192 12288 16384}"
        ;;

    query)
        shift 2
        python query.py --config "$CONFIG" "$@"
        ;;

    help|*)
        cat <<EOF
usage: ./submit.sh <task> [config] [args]

  search              trained network -> candidates, sensitivity, figures
  smoke               same, shallow background, for a first pass
  plan                show the steps and projected cost, submit nothing

  dataprep [run] [dets]      build the search-grade strain release
  dataprep-budget [run]      what it costs, fetching nothing
  dataprep-verify [run]      check a built release, checksums included

  stages              segments grid zerolag slides background far
                      injections sensitivity pastro trials candidates
                      catalogue store figdata figures tables
  chain [n]           background as n dependent jobs

  characterize ...    per-event vetting, spectrograms, follow-up, PE
  query ...           ask the campaign store a question

config defaults to config_o3a_HL
EOF
        ;;
esac
