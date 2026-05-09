#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Submit all three notebooks under root/notebooks-new/ to PBS.
#
# Usage from the cluster login node:
#   bash scripts/submit_all_new.sh             # submit all three
#   bash scripts/submit_all_new.sh loaded      # submit only loaded-weights swin-tiny
#   bash scripts/submit_all_new.sh loadedmoby     # submit only loaded-weights MOBY
#   bash scripts/submit_all_new.sh scratch     # submit only from-scratch
#   bash scripts/submit_all_new.sh cluster     # submit only clustering

#   bash scripts/submit_all_new.sh chain       # loaded -> clustering (afterok)
#
# Each individual script is independently runnable with `qsub`.
# ─────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOADED="${SCRIPT_DIR}/submit_loaded_weights.sh"
LOADEDMOBY="${SCRIPT_DIR}/submit_loaded_weights_moby.sh"
SCRATCH="${SCRIPT_DIR}/submit_from_scratch.sh"
CLUSTER="${SCRIPT_DIR}/submit_clustering.sh"

mode="${1:-all}"

submit() {
    local script="$1"
    shift
    local jobid
    jobid=$(qsub "$@" "$script")
    echo "submitted $(basename "$script")  ->  ${jobid}"
    echo "$jobid"
}

case "$mode" in
    loaded)
        submit "$LOADED" >/dev/null
        ;;
    loadedmoby)
        submit "$LOADEDMOBY" >/dev/null
        ;;
    scratch)
        submit "$SCRATCH" >/dev/null
        ;;
    cluster|clustering)
        submit "$CLUSTER" >/dev/null
        ;;
    chain)
        loaded_id=$(submit "$LOADED")
        submit "$CLUSTER" -W "depend=afterok:${loaded_id}" >/dev/null
        ;;
    all)
        submit "$LOADED" >/dev/null
        submit "$LOADEDMOBY" >/dev/null
        submit "$SCRATCH" >/dev/null
        submit "$CLUSTER" >/dev/null
        ;;
    *)
        echo "unknown mode: $mode" >&2
        echo "usage: $0 [all|loaded|loadedmoby|scratch|cluster|chain]" >&2
        exit 2
        ;;
esac
