#!/bin/bash
#SBATCH --job-name=preprocess_pseudolabels
#SBATCH --partition=default1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# Preprocess raw VSI microscopy files into normalized 128x128 patches
# with rolling-ball background subtraction (for pseudolabel generation).
# CPU-only, no GPU needed. Java/aicsimageio loads each VSI file.

PROJECT_DIR="/home/veronika.anokhina/thesis/thesis"
CONDA_ENV="microscopy_anokhver"

set -euo pipefail

mkdir -p "${PROJECT_DIR}/logs"

source ~/.bashrc
conda activate "${CONDA_ENV}"

# Cap JVM heap for Bio-Formats / aicsimageio so it doesn't balloon
# export JAVA_TOOL_OPTIONS="-Xmx16g"

echo "=== Job Info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Conda env: ${CONDA_ENV}"
echo "Start:     $(date)"
echo "================"

python "${PROJECT_DIR}/root/synaptic_ssl/utils_data/preprocess_training.py" \
    --file_extensions .vsi \
    2>&1

EXIT_CODE=$?

echo "=== Done ==="
echo "Exit code: ${EXIT_CODE}"
echo "End:       $(date)"

exit ${EXIT_CODE}
