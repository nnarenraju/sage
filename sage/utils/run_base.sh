#!/bin/bash
# ===========================================================================
# run_base.sh -- shared launch logic for Sage runs across servers.
#
# This is a *library*: source it from a thin per-run wrapper. It reads all
# server-specific values (paths, conda python, SLURM params) from the single
# source of truth in sage/utils/servers.py, so switching machines is one line:
#
#     export SAGE_SERVER=potsdam            # or leave unset to auto-detect
#     source "$(git rev-parse --show-toplevel)/sage/utils/run_base.sh"
#     sage_submit --job o3b-download "cd runs/o3b && python -c 'from dataset import make_dataset; make_dataset()'"
#
# Functions provided:
#   sage_activate            set caches / LD_LIBRARY_PATH / PATH for the env
#   sage_python              echo the interpreter that will be used
#   sage_run    "<cmd>"      run <cmd> now, in the foreground, env activated
#   sage_submit "<cmd>"      SLURM: sbatch --wrap;  local: nohup background
#                            opts: --job NAME  --time T  --gres G  --mem M  --cpus N
#                                  --nodelist NODES
# ===========================================================================

# Resolve this file's location -> repo root (.../sage/sage/utils/run_base.sh).
_SAGE_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Re-anchor to an ABSOLUTE path: sage_submit embeds this into the sbatch --wrap
# and the job re-sources it from the compute node's cwd, where a relative path
# (e.g. when this file was sourced as "sage/utils/run_base.sh") would not exist.
_SAGE_RUN_BASE="$_SAGE_UTILS_DIR/run_base.sh"
SAGE_REPO_ROOT="${SAGE_REPO_ROOT:-$(cd "$_SAGE_UTILS_DIR/../.." && pwd)}"

# Pick a python to drive the (stdlib-only) server registry bridge. Any python
# works here; it does not need the sage deps.
_sage_bridge_python() {
    if [ -n "$SAGE_PYTHON" ] && [ -x "$SAGE_PYTHON" ]; then echo "$SAGE_PYTHON"; return; fi
    command -v python3 || command -v python
}

# Import every SAGE_* value from servers.py into this shell.
sage_load_server() {
    local exports
    exports="$(PYTHONPATH="$SAGE_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        "$(_sage_bridge_python)" -m sage.utils.servers env)" || {
        echo "[run_base] failed to resolve server (SAGE_SERVER='${SAGE_SERVER:-}')." >&2
        return 1
    }
    eval "$exports"
}

# Prepare the runtime environment for an activated server.
sage_activate() {
    [ -n "$SAGE_DATA_ROOT" ] || sage_load_server || return 1

    # Put the env's python first on PATH so `python` == the sage conda env.
    if [ -n "$SAGE_PYTHON" ] && [ -x "$SAGE_PYTHON" ]; then
        export PATH="$(dirname "$SAGE_PYTHON"):$PATH"
        export PY="$SAGE_PYTHON"
    else
        export PY="$(command -v python3 || command -v python)"
    fi

    # Expose pip-installed NVIDIA shared libs (libnvrtc.so.12 etc.).
    local site nvidia
    site="$("$PY" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)"
    nvidia="$site/nvidia"
    if [ -d "$nvidia" ]; then
        export LD_LIBRARY_PATH="$(find "$nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"
    fi

    # Redirect EVERY cache/temp away from $HOME (50 GB hard cap) and /tmp, onto
    # work storage. Covers torch/triton/hf plus the usual offenders that quietly
    # fill home: astropy (IERS/leap-second downloads via XDG_CACHE_HOME),
    # matplotlib font cache, numba, and generic XDG + TMPDIR scratch.
    local cache="$SAGE_WORK_ROOT/cache"
    export TORCH_HOME="$cache/torch"
    export TRITON_CACHE_DIR="$cache/triton"
    export HF_HOME="$cache/huggingface"
    export XDG_CACHE_HOME="$cache/xdg"            # astropy & many libs honour this
    export XDG_CONFIG_HOME="$cache/xdg-config"
    export MPLCONFIGDIR="$cache/matplotlib"
    export NUMBA_CACHE_DIR="$cache/numba"
    export ASTROPY_CACHE_DIR="$cache/astropy"     # explicit, in case XDG is ignored
    export TMPDIR="$SAGE_WORK_ROOT/tmp"           # keep scratch off /tmp and home
    export PYTHONPATH="$SAGE_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" \
             "$XDG_CONFIG_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" \
             "$ASTROPY_CACHE_DIR" "$TMPDIR"
}

sage_python() { sage_activate >/dev/null 2>&1; echo "$PY"; }

# Run a command now, in the foreground.
sage_run() {
    sage_activate || return 1
    echo "[run_base] server=$SAGE_SERVER  data_root=$SAGE_DATA_ROOT  python=$PY"
    bash -c "$*"
}

# Submit a command: SLURM via `sbatch --wrap`, or run locally in the background.
# All SLURM flags come from the server registry; per-call --opts override them.
sage_submit() {
    sage_activate || return 1

    local job="sage" part="$SAGE_PARTITION" qos="$SAGE_QOS"
    local time="$SAGE_TIME" gres="$SAGE_GRES" mem="$SAGE_MEM" cpus="$SAGE_CPUS"
    # Campaign-level pins, settable from the environment so the per-run submit.sh
    # needs no edit; the matching --opts override them per call.
    #
    # Prefer SAGE_GRES_OVERRIDE (a typed GRES, e.g. gpu:h100_80gb:1) over
    # SAGE_NODELIST to confine a campaign to a subset of the GPUs. A typed GRES is
    # a consumable the scheduler must actually reserve, so surplus jobs queue.
    # --nodelist only constrains PLACEMENT: asking for more GPUs than the named
    # node has was observed to place the extra jobs on it with GRES=(null) -- they
    # start with no GPU and die on cuda:0. Use --nodelist only when the node's GPU
    # count is not the binding constraint.
    local dep="" parsable="" array="" nodelist="${SAGE_NODELIST:-}"
    [ -n "${SAGE_GRES_OVERRIDE:-}" ] && gres="$SAGE_GRES_OVERRIDE"
    while [ "$#" -gt 1 ]; do
        case "$1" in
            --job)       job="$2";  shift 2 ;;
            --partition) part="$2"; shift 2 ;;
            --qos)       qos="$2";  shift 2 ;;
            --time)      time="$2"; shift 2 ;;
            # Pass --gres "" (or "none") to drop it, e.g. for CPU partitions.
            --gres)      gres="$2"; [ "$gres" = "none" ] && gres=""; shift 2 ;;
            --mem)       mem="$2";  shift 2 ;;
            --cpus)      cpus="$2"; shift 2 ;;
            # Restrict the job to specific node(s), e.g. --nodelist hgc01. Used to
            # partition the GPU pool between concurrent campaigns so one does not
            # crowd the other off the machine.
            --nodelist)  nodelist="$2"; shift 2 ;;
            # SLURM job dependency, e.g. "afterany:12345" (used by chaining).
            --dependency) dep="$2"; shift 2 ;;
            # SLURM array spec, e.g. "1-6%6" -- six tasks, six at a time. Used to
            # spread one stage across several GPUs. Without this the flag fell
            # through to the command position, so the array spec was submitted as
            # the job and every task exited 127.
            --array)     array="$2"; shift 2 ;;
            # Print ONLY the job id on stdout (for capturing in a chain driver);
            # all human-readable log lines go to stderr.
            --parsable)  parsable=1; shift ;;
            *) break ;;
        esac
    done
    local cmd="$1"
    [ -n "$cmd" ] || { echo "[run_base] sage_submit: no command given" >&2; return 2; }

    if [ "$SAGE_SCHEDULER" = "slurm" ]; then
        # The job re-sources this library so it is fully self-contained; the
        # submitting environment (SAGE_SERVER etc.) is also inherited by SLURM.
        local wrap="export SAGE_SERVER='$SAGE_SERVER'; source '$_SAGE_RUN_BASE'; sage_activate; cd '$PWD'; $cmd"
        local args=(--job-name="$job" --chdir="$PWD" --nodes=1 --ntasks=1
                    --cpus-per-task="$cpus" --output="%x-%j.out"
                    --mail-type=ALL)
        [ -n "$part" ]           && args+=(--partition="$part")
        [ -n "$qos" ]            && args+=(--qos="$qos")
        [ -n "$gres" ]           && args+=(--gres="$gres")
        [ -n "$time" ]           && args+=(--time="$time")
        [ -n "$mem" ]            && args+=(--mem="$mem")
        [ -n "$dep" ]            && args+=(--dependency="$dep")
        [ -n "$array" ]          && args+=(--array="$array")
        [ -n "$nodelist" ]       && args+=(--nodelist="$nodelist")
        [ -n "$parsable" ]       && args+=(--parsable)
        [ -n "$SAGE_ACCOUNT" ]   && args+=(--account="$SAGE_ACCOUNT")
        [ -n "$SAGE_MAIL" ]      && args+=(--mail-user="$SAGE_MAIL")
        # stderr, so `--parsable` leaves only the job id on stdout.
        echo "[run_base] sbatch ${args[*]}" >&2
        sbatch "${args[@]}" --wrap "$wrap"
    else
        local log="$job-$(date +%Y%m%d-%H%M%S).out"
        echo "[run_base] local run (scheduler=$SAGE_SCHEDULER) -> $log"
        nohup bash -c "$cmd" >"$log" 2>&1 &
        echo "[run_base] pid $! ; tail -f $log"
    fi
}

# Submit <n> copies of a command as an `afterany` dependency chain, so a long
# run spans the scheduler's per-job wall limit as N back-to-back segments.
#   sage_submit_chain 4 --time 2-00:00 --job o3b-hard "<cmd>"
# Each segment starts only after the previous one ENDS (wall-kill or normal
# exit) and resumes from its own checkpoint; once training completes, trailing
# segments resume, find no epochs left, and exit in seconds. Only ONE segment
# ever runs at a time (serial chain) -> holds at most one GPU. Echoes the last
# segment's job id.
sage_submit_chain() {
    local n="$1"; shift
    case "$n" in ''|*[!0-9]*) echo "[run_base] sage_submit_chain: first arg must be N>=1" >&2; return 2 ;; esac
    [ "$n" -ge 1 ] || { echo "[run_base] sage_submit_chain: N must be >=1" >&2; return 2; }
    local prev="" jid dep_arg
    local i
    for i in $(seq 1 "$n"); do
        dep_arg=()
        [ -n "$prev" ] && dep_arg=(--dependency "afterany:$prev")
        jid="$(sage_submit --parsable "${dep_arg[@]}" "$@")" || return 1
        jid="${jid%%;*}"        # strip ";cluster" if the site returns it
        echo "[run_base] chain segment $i/$n -> job $jid${prev:+ (afterany:$prev)}" >&2
        prev="$jid"
    done
    echo "$prev"
}

# Load the server now so sourcing scripts can read $SAGE_* immediately.
sage_load_server || true
