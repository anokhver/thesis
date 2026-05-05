#!/bin/bash
# Submits AE, SimMIM, MAE pretrains as separate PBS jobs.
# Each job gets its own GPU and runs in parallel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMITTER="${SCRIPT_DIR}/submit_pretrain_loaded.sh"

NOTEBOOKS=(
    # "pretrain_ea_swin.ipynb"
    "pretrain_simmim_vicreg_swin.ipynb"

)

for nb in "${NOTEBOOKS[@]}"; do
    # cleaner PBS job name: drop the leading "0N_" and the trailing ".ipynb"
    name="${nb%.ipynb}"
    name="${name#[0-9][0-9]_}"
    echo "submitting ${nb} as PBS job '${name}'..."
    qsub -N "${name}" -v "NOTEBOOK=${nb}" "${SUBMITTER}"
done

echo
echo "track with: qstat -u \$USER"
