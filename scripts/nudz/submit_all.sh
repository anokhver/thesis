#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Submit all three notebooks under notebooks/ to Slurm (NUDZ).
#
# Migrated from submit_all.sh (PBS qsub -> Slurm sbatch).
#
# Usage from the cluster login node:
#   bash scripts/nudz/submit_all.sh             # submit all three
#   bash scripts/nudz/submit_all.sh loadedmoby  # submit only loaded-weights MOBY
#   bash scripts/nudz/submit_all.sh scratch     # submit only from-scratch
#   bash scripts/nudz/submit_all.sh cluster     # submit only clustering
#   bash scripts/nudz/submit_all.sh chain       # loadedmoby -> clustering (afterok)
#
# Each individual script is independently runnable with `sbatch`.
# ─────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOADEDMOBY="${SCRIPT_DIR}/submit_loaded_moby.slurm.sh"
SCRATCH="${SCRIPT_DIR}/submit_from_scratch.slurm.sh"
CLUSTER="${SCRIPT_DIR}/submit_clustering.slurm.sh"

mode="${1:-all}"

# sbatch --parsable returns just the numeric job id (optionally cluster name);
# strip any ";cluster" suffix to keep the bare id for dependency strings.
submit() {
    local script="$1"
    shift
    local jobid
    jobid=$(sbatch --parsable "$@" "$script")
    jobid="${jobid%%;*}"
    echo "submitted $(basename "$script")  ->  ${jobid}" >&2
    echo "$jobid"
}

case "$mode" in
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
        loaded_id=$(submit "$LOADEDMOBY")
        submit "$CLUSTER" --dependency="afterok:${loaded_id}" >/dev/null
        ;;
    all)
        submit "$LOADEDMOBY" >/dev/null
        submit "$SCRATCH" >/dev/null
        submit "$CLUSTER" >/dev/null
        ;;
    *)
        echo "unknown mode: $mode" >&2
        echo "usage: $0 [all|loadedmoby|scratch|cluster|chain]" >&2
        exit 2
        ;;
esac
