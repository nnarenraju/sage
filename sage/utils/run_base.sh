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
# ===========================================================================

# Resolve this file's location -> repo root (.../sage/sage/utils/run_base.sh).
_SAGE_RUN_BASE="${BASH_SOURCE[0]}"
_SAGE_UTILS_DIR="$(cd "$(dirname "$_SAGE_RUN_BASE")" && pwd)"
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
        [ -n "$SAGE_ACCOUNT" ]   && args+=(--account="$SAGE_ACCOUNT")
        [ -n "$SAGE_MAIL" ]      && args+=(--mail-user="$SAGE_MAIL")
        echo "[run_base] sbatch ${args[*]}"
        sbatch "${args[@]}" --wrap "$wrap"
    else
        local log="$job-$(date +%Y%m%d-%H%M%S).out"
        echo "[run_base] local run (scheduler=$SAGE_SCHEDULER) -> $log"
        nohup bash -c "$cmd" >"$log" 2>&1 &
        echo "[run_base] pid $! ; tail -f $log"
    fi
}

# Load the server now so sourcing scripts can read $SAGE_* immediately.
sage_load_server || true
