#!/bin/bash
#PBS -N encoder_audit
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_mem=16000mb:scratch_local=10gb
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/
#PBS -m abe
#PBS -M veronika.i.anokhina@gmail.com

# Run encoder/projector feature-quality audit across all checkpoints under
# RUNS_ROOT, writing results to RUNS_ROOT/encoder_audit/ (does not touch
# individual run folders).
#
# Usage:
#   qsub scripts/metacentrum/submit_encoder_audit.sh
#   qsub -v RUNS_ROOT=/path/to/other_outputs scripts/metacentrum/submit_encoder_audit.sh

PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
RUNS_ROOT="${RUNS_ROOT:-${PROJECT_DIR}/data/training_outputs_not_all_excluded}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data/patches_128_from_zip}"
CKPT_NAME="${CKPT_NAME:-best_model.pt}"
N_SAMPLES="${N_SAMPLES:-8000}"
CONDA_ENV="microscopy"

set -euo pipefail

mkdir -p "${PROJECT_DIR}/logs"

source /cvmfs/software.metacentrum.cz/conda/envs/miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate /storage/brno2/home/anokhver/.conda/envs/${CONDA_ENV}

cd "${PROJECT_DIR}"

echo "=== Job Info ==="
echo "Job ID:      ${PBS_JOBID:-local}"
echo "Node:        $(hostname)"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Conda env:   ${CONDA_ENV}"
echo "Runs root:   ${RUNS_ROOT}"
echo "Data root:   ${DATA_ROOT}"
echo "Checkpoint:  ${CKPT_NAME}"
echo "N samples:   ${N_SAMPLES}"
echo "Start:       $(date)"
echo "================"

python scripts/encoder_feature_audit.py \
    --runs_root  "${RUNS_ROOT}" \
    --data_root  "${DATA_ROOT}" \
    --out_dir    "${RUNS_ROOT}/encoder_audit" \
    --ckpt_name  "${CKPT_NAME}" \
    --n_samples  "${N_SAMPLES}" \
    --device     cuda \
    2>&1

EXIT_CODE=$?

echo "=== Done ==="
echo "Exit code: ${EXIT_CODE}"
echo "End:       $(date)"

exit ${EXIT_CODE}
