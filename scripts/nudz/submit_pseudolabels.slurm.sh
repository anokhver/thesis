#!/bin/bash
#SBATCH --job-name=pseudolabels
#SBATCH --partition=default1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=veronika.i.anokhina@gmail.com
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Migrated from submit_pseudolabels.sh (MetaCentrum PBS Pro -> NUDZ Slurm)
# Changes: CPU-only -> default1 partition, env activation via ~/.bashrc,
# project dir under /home/veronika.anokhina, conda env microscopy_anokhver.

# Generate blob pseudo-labels for synaptic puncta (CPU-only, no GPU needed).

PROJECT_DIR="/home/veronika.anokhina/thesis/thesis"
NOTEBOOK_DIR="${PROJECT_DIR}/notebooks/pseudolabels"
NOTEBOOK="blob_pseudolabels.ipynb"
RUN_DIR="${NOTEBOOK_DIR}/runs"
CONDA_ENV="microscopy_anokhver"

set -euo pipefail

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${RUN_DIR}"

source ~/.bashrc
conda activate "${CONDA_ENV}"

cd "${NOTEBOOK_DIR}"

echo "=== Job Info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Conda env: ${CONDA_ENV}"
echo "Notebook:  ${NOTEBOOK_DIR}/${NOTEBOOK}"
echo "Start:     $(date)"
echo "================"

OUTPUT_NOTEBOOK="${RUN_DIR}/pseudolabels_RUN_$(date +%Y%m%d_%H%M%S).ipynb"

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
