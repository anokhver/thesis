#!/bin/bash
#PBS -N loaded_weights
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_mem=40gb:scratch_local=50gb:cluster=zia
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/
#PBS -m abe
#PBS -M veronika.i.anokhina@gmail.com

# Continuation pretrain (timm-ImageNet Swin-Tiny init) with SimMIM + VICReg.

PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
NOTEBOOK_DIR="${PROJECT_DIR}/root/notebooks/loaded-weights"
NOTEBOOK="pretrain_simmim_vicreg_moby.ipynb"
RUN_DIR="${NOTEBOOK_DIR}/runs"
CONDA_ENV="microscopy"

set -euo pipefail

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${RUN_DIR}"

source /cvmfs/software.metacentrum.cz/conda/envs/miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate /storage/brno2/home/anokhver/.conda/envs/${CONDA_ENV}

cd "${NOTEBOOK_DIR}"

echo "=== Job Info ==="
echo "Job ID:    ${PBS_JOBID}"
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
