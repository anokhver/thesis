#!/bin/bash
#PBS -N train_joint_2ch
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_mem=40000mb:scratch_local=20gb
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/
#PBS -m abe
#PBS -M veronika.i.anokhina@gmail.com

# Joint 2-channel (PRE+POST) SwinUNETR fine-tuning from a SimMIM+VICReg
# pretrained encoder. Outputs go to data/training_outputs/joint_2ch/.
#
# Override the config (default: joint_2ch_default.json) with:
#   qsub scripts/metacentrum/submit_train_joint_2ch.sh
#   qsub -v CONFIG=configs/segmentation/joint_2ch_smoke.json \
#        scripts/metacentrum/submit_train_joint_2ch.sh

PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/segmentation/joint_2ch_default.json}"
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

python scripts/training/train_swinunetr_joint_2ch.py \
    --config "${CONFIG}" \
    2>&1

EXIT_CODE=$?

echo "=== Done ==="
echo "Exit code: ${EXIT_CODE}"
echo "End:       $(date)"

exit ${EXIT_CODE}
