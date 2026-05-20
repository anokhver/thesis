#!/bin/bash
#PBS -N recover_post_training
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_mem=24000mb:scratch_local=10gb
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/
#PBS -m abe
#PBS -M veronika.i.anokhina@gmail.com

# Regenerate post-training visualisations (curves + recon panels + encoder
# export) for every run folder under OUTPUT_ROOT. Recursive: handles the
# grouped layout pretrain_{moby,scratch,tinny22k}/<run>/.
#
# Usage:
#   qsub scripts/metacentrum/submit_recover_post_training.sh
#   qsub -v OUTPUT_ROOT=/storage/brno2/home/anokhver/thesis/data/training_outputs/pretrain_moby \
#        scripts/metacentrum/submit_recover_post_training.sh
#   qsub -v RUN_FULL_IMAGE=1 scripts/metacentrum/submit_recover_post_training.sh

PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/data/training_outputs}"
RUN_FULL_IMAGE="${RUN_FULL_IMAGE:-0}"
CONDA_ENV="microscopy"

set -euo pipefail

mkdir -p "${PROJECT_DIR}/logs"

source /cvmfs/software.metacentrum.cz/conda/envs/miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate /storage/brno2/home/anokhver/.conda/envs/${CONDA_ENV}

cd "${PROJECT_DIR}"

echo "=== Job Info ==="
echo "Job ID:        ${PBS_JOBID:-local}"
echo "Node:          $(hostname)"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Conda env:     ${CONDA_ENV}"
echo "Output root:   ${OUTPUT_ROOT}"
echo "Full image:    ${RUN_FULL_IMAGE}"
echo "Start:         $(date)"
echo "================"

EXTRA_FLAGS=""
if [[ "${RUN_FULL_IMAGE}" == "1" ]]; then
    EXTRA_FLAGS="--run-full-image"
fi

python scripts/training/recover_post_training_viz.py \
    --output-root "${OUTPUT_ROOT}" \
    --device auto \
    ${EXTRA_FLAGS} \
    2>&1

EXIT_CODE=$?

echo "=== Done ==="
echo "Exit code: ${EXIT_CODE}"
echo "End:       $(date)"

exit ${EXIT_CODE}
