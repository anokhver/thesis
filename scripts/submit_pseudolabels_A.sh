#!/bin/bash
#PBS -N pseudolabels_A
#PBS -l select=1:ncpus=4:mem=16gb:scratch_local=10gb
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/

# ── Pseudo-label generation: Pipeline A ──────────────────────
# Runs LoG blob detection on all channels, filters by foreground mask.
# CPU only — no GPU needed.
# ─────────────────────────────────────────────────────────────

PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
NOTEBOOK_DIR="${PROJECT_DIR}/notebooks/training"
NOTEBOOK="generate_pseudolabels_A.ipynb"
RUN_DIR="${NOTEBOOK_DIR}/runs"
CONDA_ENV="microscopy"

# ── Setup ────────────────────────────────────────────────────
set -euo pipefail

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${RUN_DIR}"

source /cvmfs/software.metacentrum.cz/conda/envs/miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate /storage/brno2/home/anokhver/.conda/envs/${CONDA_ENV}

# cd into NOTEBOOK_DIR (NOT RUN_DIR) so notebook imports resolve correctly.
# papermill writes the executed copy to RUN_DIR but the kernel's CWD stays here.
cd "${NOTEBOOK_DIR}"

echo "=== Job Info ==="
echo "Job ID:    ${PBS_JOBID}"
echo "Node:      $(hostname)"
echo "Conda env: ${CONDA_ENV}"
echo "Notebook:  ${NOTEBOOK_DIR}/${NOTEBOOK}"
echo "Start:     $(date)"
echo "================"

# ── Run notebook ─────────────────────────────────────────────
OUTPUT_NOTEBOOK="${RUN_DIR}/generate_pseudolabels_A_RUN_$(date +%Y%m%d_%H%M%S).ipynb"

python -m papermill \
    "${NOTEBOOK}" \
    "${OUTPUT_NOTEBOOK}" \
    --no-progress-bar \
    --log-output \
    --request-save-on-cell-execute \
    2>&1

EXIT_CODE=$?

echo "=== Done ==="
echo "Exit code: ${EXIT_CODE}"
echo "End:       $(date)"
echo "Output:    ${OUTPUT_NOTEBOOK}"

exit ${EXIT_CODE}
