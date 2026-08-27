#!/usr/bin/env bash
#
# Submit a sequence of stages as one dependent chain.
#
#   ./chain_stages.sh <config> <stage> [stage ...]
#
# Each job holds until the previous one succeeds. Every stage is forced, because the
# reason to run a chain by hand is that a code change invalidated products the manifest
# still records as complete -- and the spec hash is deliberately blind to code.
#
# The job id is captured and *checked*. Losing it silently is not a cosmetic failure: the
# next submission then carries no --dependency and starts immediately, so a downstream
# stage reads the products of the run it was supposed to follow. That has happened twice.
set -euo pipefail

cd "$(dirname "$0")"

CONFIG="${1:?usage: ./chain_stages.sh <config> <stage> [stage ...]}"
shift
[ "$#" -gt 0 ] || { echo "no stages given" >&2; exit 2; }

AFTER=""
for STAGE in "$@"; do
    OUT=$(SAGE_FORCE=1 SAGE_AFTER="$AFTER" ./submit.sh "$STAGE" "$CONFIG" 2>&1) || {
        echo "$OUT" >&2
        echo "chain: submitting '$STAGE' failed" >&2
        exit 1
    }
    JOB=$(printf '%s\n' "$OUT" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -1)
    if [ -z "$JOB" ]; then
        echo "$OUT" >&2
        echo "chain: no job id from '$STAGE' -- refusing to submit the rest without a" >&2
        echo "       dependency, which would let a later stage read stale products." >&2
        [ -n "$AFTER" ] && echo "       cancel what is already queued: scancel $AFTER" >&2
        exit 1
    fi
    printf '%-12s %s%s\n' "$STAGE" "$JOB" "${AFTER:+  (after $AFTER)}"
    AFTER="$JOB"
done
