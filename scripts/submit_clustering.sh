#!/bin/bash
#PBS -N clustering
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_mem=24gb:scratch_local=20gb
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/
#PBS -m abe
#PBS -M your@email.com

# Clustering / representation-quality check on a trained encoder.
# Inference only -- modest GPU is enough; no training, short walltime.

PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
NOTEBOOK_DIR="${PROJECT_DIR}/notebooks/clustering"
NOTEBOOK="clustering_check.ipynb"
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

OUTPUT_NOTEBOOK="${RUN_DIR}/clustering_RUN_$(date +%Y%m%d_%H%M%S).ipynb"

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
