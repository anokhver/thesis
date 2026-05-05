#!/bin/bash
#PBS -N pretrain_v2
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_mem=40gb:scratch_local=50gb:cluster=zia
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/
#PBS -m abe
#PBS -M your@email.com

# Generic v2 pretraining submitter. Runs whatever notebook is passed in via
# `qsub -v NOTEBOOK=<file.ipynb>`. PBS -N is overridden by the wrapper too.

set -euo pipefail

PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
NOTEBOOK_DIR="${PROJECT_DIR}/notebooks/loaded-weiights-pretrained/"
RUN_DIR="${NOTEBOOK_DIR}/runs"
CONDA_ENV="microscopy"

: "${NOTEBOOK:?must be passed via qsub -v NOTEBOOK=<filename.ipynb>}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${RUN_DIR}"

source /cvmfs/software.metacentrum.cz/conda/envs/miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate /storage/brno2/home/anokhver/.conda/envs/${CONDA_ENV}

# stay in NOTEBOOK_DIR so notebooks' relative imports resolve
cd "${NOTEBOOK_DIR}"

echo "=== Job Info ==="
echo "Job ID:    ${PBS_JOBID:-interactive}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Conda env: ${CONDA_ENV}"
echo "Notebook:  ${NOTEBOOK_DIR}/${NOTEBOOK}"
echo "Start:     $(date)"
echo "================"

stem="${NOTEBOOK%.ipynb}"
OUTPUT_NOTEBOOK="${RUN_DIR}/${stem}_RUN_$(date +%Y%m%d_%H%M%S).ipynb"

set +e
python -m papermill \
    "${NOTEBOOK}" \
    "${OUTPUT_NOTEBOOK}" \
    --no-progress-bar \
    --log-output \
    --request-save-on-cell-execute \
    2>&1
EXIT_CODE=$?
set -e

echo "=== Done ==="
echo "Exit code: ${EXIT_CODE}"
echo "End:       $(date)"
echo "Output:    ${OUTPUT_NOTEBOOK}"

exit ${EXIT_CODE}
