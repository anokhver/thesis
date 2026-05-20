#!/bin/bash
#PBS -N tile_from_mip_zip
#PBS -l select=1:ncpus=8:mem=64gb:scratch_local=20gb
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/
#PBS -m abe
#PBS -M veronika.i.anokhina@gmail.com

# Tile full-MIP .npy files stored inside a .zip archive into 128x128
# patches without extracting the archive.
#
# Submit with overrides:
#   qsub -v ZIP_PATH=/storage/.../Microscopy.zip,OUTPUT_ROOT=/storage/.../patches_128_from_zip,WORKERS=8 \
#        scripts/metacentrum/submit_tile_from_mip_zip.sh
#
# Or just edit the defaults below and run:
#   qsub scripts/metacentrum/submit_tile_from_mip_zip.sh

PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
CONDA_ENV="microscopy"

# Defaults (override with `qsub -v KEY=VAL,...`)
: "${ZIP_PATH:=${PROJECT_DIR}/data/Microscopy.zip}"
: "${OUTPUT_ROOT:=${PROJECT_DIR}/data/patches_128_from_zip}"
: "${PATCH_SIZE:=128}"
: "${WORKERS:=8}"        # parallel processes per date folder (0 = auto)
: "${EXTRA_ARGS:=}"       # e.g. "--date 20251030" or "--max_folders 1 --dry_run"

set -euo pipefail

mkdir -p "${PROJECT_DIR}/logs"

source /cvmfs/software.metacentrum.cz/conda/envs/miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate /storage/brno2/home/anokhver/.conda/envs/${CONDA_ENV}

echo "=== Job Info ==="
echo "Job ID:       ${PBS_JOBID:-<interactive>}"
echo "Node:         $(hostname)"
echo "Conda env:    ${CONDA_ENV}"
echo "Zip:          ${ZIP_PATH}"
echo "Output root:  ${OUTPUT_ROOT}"
echo "Patch size:   ${PATCH_SIZE}"
echo "Workers:      ${WORKERS}"
echo "Extra args:   ${EXTRA_ARGS}"
echo "Start:        $(date)"
echo "================"

cd "${PROJECT_DIR}"

python "${PROJECT_DIR}/scripts/preprocess/tile_from_mip.py" \
    --zip         "${ZIP_PATH}" \
    --output_root "${OUTPUT_ROOT}" \
    --patch_size  "${PATCH_SIZE}" \
    --workers     "${WORKERS}" \
    ${EXTRA_ARGS} \
    2>&1

EXIT_CODE=$?

echo "=== Done ==="
echo "Exit code: ${EXIT_CODE}"
echo "End:       $(date)"

exit ${EXIT_CODE}
