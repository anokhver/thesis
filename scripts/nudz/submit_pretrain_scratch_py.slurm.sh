#!/bin/bash
#SBATCH --job-name=pretrain_scratch_script
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

# SimMIM+VICReg pretrain from scratch (script version).
# Outputs go to outputs/ at the repo root.

PROJECT_DIR="/home/veronika.anokhina/thesis/thesis"
CONFIG="${PROJECT_DIR}/configs/pretrain_scratch/default.json"
CONDA_ENV="microscopy_anokhver"

set -euo pipefail

mkdir -p "${PROJECT_DIR}/logs"

source ~/.bashrc
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

echo "=== Job Info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
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
