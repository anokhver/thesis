#!/bin/bash
#PBS -N preprocess_pseudolabels
#PBS -l select=1:ncpus=8:mem=128gb:scratch_local=50gb
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/
#PBS -m abe
#PBS -M veronika.i.anokhina@gmail.com

# Preprocess raw VSI microscopy files into normalized 128x128 patches
# with rolling-ball background subtraction (for pseudolabel generation).
# CPU-only, no GPU needed. Java/aicsimageio loads each VSI file.

PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
CONDA_ENV="microscopy"

set -euo pipefail

mkdir -p "${PROJECT_DIR}/logs"

source /cvmfs/software.metacentrum.cz/conda/envs/miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate /storage/brno2/home/anokhver/.conda/envs/${CONDA_ENV}

# Cap JVM heap for Bio-Formats / aicsimageio so it doesn't balloon
# export JAVA_TOOL_OPTIONS="-Xmx16g"

echo "=== Job Info ==="
# echo "Job ID:    ${PBS_JOBID}"
echo "Node:      $(hostname)"
echo "Conda env: ${CONDA_ENV}"
echo "Start:     $(date)"
echo "================"

python "${PROJECT_DIR}/root/utils_data/preprocess_pseudolabels.py" \
    --file_extensions .vsi \
    2>&1

EXIT_CODE=$?

echo "=== Done ==="
echo "Exit code: ${EXIT_CODE}"
echo "End:       $(date)"

exit ${EXIT_CODE}
