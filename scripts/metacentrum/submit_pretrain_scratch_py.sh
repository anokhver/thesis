#!/bin/bash
#PBS -N pretrain_scratch_script
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_mem=95000mb:scratch_local=20gb
#PBS -l walltime=10:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/
#PBS -m abe
#PBS -M veronika.i.anokhina@gmail.com

# SimMIM+VICReg pretrain from scratch (script version).
# Outputs go to data/training_outputs/.
#
# Variant selection (default: 128_default.json):
#   qsub scripts/submit_pretrain_scratch_py.sh
#   qsub -v CONFIG=configs/pretrain_scratch/128_no_fourier_vicreg_on.json  scripts/submit_pretrain_scratch_py.sh
#   qsub -v CONFIG=configs/pretrain_scratch/128_no_fourier_vicreg_off.json scripts/submit_pretrain_scratch_py.sh

PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/pretrain_scratch/128_default.json}"
CONDA_ENV="microscopy"

set -euo pipefail

mkdir -p "${PROJECT_DIR}/logs"

source /cvmfs/software.metacentrum.cz/conda/envs/miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate /storage/brno2/home/anokhver/.conda/envs/${CONDA_ENV}

cd "${PROJECT_DIR}"

echo "=== Job Info ==="
echo "Job ID:    ${PBS_JOBID}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Conda env: ${CONDA_ENV}"
echo "Config:    ${CONFIG}"
echo "Start:     $(date)"
echo "================"

python scripts/pretrain_simmim_vicreg.py \
    --config "${CONFIG}" \
    2>&1

EXIT_CODE=$?

echo "=== Done ==="
echo "Exit code: ${EXIT_CODE}"
echo "End:       $(date)"

exit ${EXIT_CODE}
