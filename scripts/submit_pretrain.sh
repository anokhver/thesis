#!/bin/bash
#PBS -N simmim_pretrain
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_mem=40gb:scratch_local=50gb:cluster=zia
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/

# ── Paths ────────────────────────────────────────────────────
PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
NOTEBOOK_DIR="${PROJECT_DIR}/notebooks/training"
NOTEBOOK="pretrain_simMIM_swin_v2_fixed.ipynb"
RUN_DIR="${NOTEBOOK_DIR}/runs"
CONDA_ENV="microscopy"

# ── Setup ────────────────────────────────────────────────────
set -euo pipefail

# Create log + run output directories if missing
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${RUN_DIR}"

# Activate conda using full env path (works on any MetaCentrum cluster)
source /cvmfs/software.metacentrum.cz/conda/envs/miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate /storage/brno2/home/anokhver/.conda/envs/${CONDA_ENV}

# stay in NOTEBOOK_DIR so notebooks' relative imports (../..) still resolve
cd "${NOTEBOOK_DIR}"

echo "=== Job Info ==="
echo "Job ID:    ${PBS_JOBID}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Conda env: ${CONDA_ENV}"
echo "Notebook:  ${NOTEBOOK_DIR}/${NOTEBOOK}"
echo "Start:     $(date)"
echo "================"

# ── Run notebook ─────────────────────────────────────────────
# papermill executes notebooks cell-by-cell and saves output to a
# separate file so the original stays clean.
# Install once: pip install papermill
OUTPUT_NOTEBOOK="${RUN_DIR}/pretrain_simMIM_swin_RUN_$(date +%Y%m%d_%H%M%S).ipynb"

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
