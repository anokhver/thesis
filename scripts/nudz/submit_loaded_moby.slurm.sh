#!/bin/bash
#SBATCH --job-name=loaded_weights
#SBATCH --partition=ESO_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Migrated from submit_loaded_moby.sh (MetaCentrum PBS Pro -> NUDZ Slurm)
# Changes: partition gpu -> ESO_gpu (single H100), env activation via ~/.bashrc,
# project dir under /home/veronika.anokhina, conda env microscopy_anokhver.

# Loaded-weights pretrain (starting from pre-trained weights) with SimMIM + VICReg.
# Shorter walltime than from_scratch because epoch budget is 1x.

PROJECT_DIR="/home/veronika.anokhina/thesis/thesis"
NOTEBOOK_DIR="${PROJECT_DIR}/notebooks/loaded-weights"
NOTEBOOK="pretrain_simmim_vicreg_moby.ipynb"
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
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Conda env: ${CONDA_ENV}"
echo "Notebook:  ${NOTEBOOK_DIR}/${NOTEBOOK}"
echo "Start:     $(date)"
echo "================"

OUTPUT_NOTEBOOK="${RUN_DIR}/loaded_weights_RUN_$(date +%Y%m%d_%H%M%S).ipynb"

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
