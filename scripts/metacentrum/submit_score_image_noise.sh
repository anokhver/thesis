#!/bin/bash
#PBS -N score_image_noise
#PBS -l select=1:ncpus=4:mem=32gb:scratch_local=10gb
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/
#PBS -m abe
#PBS -M veronika.i.anokhina@gmail.com

# Bulk-score every source image under a patches root with the
# frequency-domain noise detector. Produces a per-image CSV and a
# JSON denylist of source filenames to feed into
# ``PatchDataset(exclude_sources=...)``.
#
# Submit with overrides:
#   qsub -v PATCH_ROOT=/storage/.../patches_128_from_zip,\
#OUT_CSV=data/noise_scores.csv,OUT_JSON=data/flagged_sources.json,\
#HP_THRESH=0.55,HF_THRESH=0.50,TOP_PERCENTILE=15 \
#        scripts/metacentrum/submit_score_image_noise.sh
#
# Or just run with the defaults below:
#   qsub scripts/metacentrum/submit_score_image_noise.sh

PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
CONDA_ENV="microscopy"

# Defaults (override with `qsub -v KEY=VAL,...`)
: "${PATCH_ROOT:=${PROJECT_DIR}/data/patches_128_from_zip}"
: "${OUT_CSV:=${PROJECT_DIR}/data/noise_scores.csv}"
: "${OUT_JSON:=${PROJECT_DIR}/data/flagged_sources.json}"
: "${EXCLUDE_PATTERNS:=KONTROLA}"
: "${HP_THRESH:=0.55}"
: "${HF_THRESH:=0.50}"
: "${TOP_PERCENTILE:=}"     # leave empty to use only the absolute thresholds
: "${EXTRA_ARGS:=}"

set -euo pipefail

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "$(dirname "${OUT_CSV}")" "$(dirname "${OUT_JSON}")"

source /cvmfs/software.metacentrum.cz/conda/envs/miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate /storage/brno2/home/anokhver/.conda/envs/${CONDA_ENV}

echo "=== Job Info ==="
echo "Job ID:           ${PBS_JOBID:-<interactive>}"
echo "Node:             $(hostname)"
echo "Conda env:        ${CONDA_ENV}"
echo "Patch root:       ${PATCH_ROOT}"
echo "Out CSV:          ${OUT_CSV}"
echo "Out JSON:         ${OUT_JSON}"
echo "Exclude patterns: ${EXCLUDE_PATTERNS}"
echo "hp_var_ratio  >=  ${HP_THRESH}"
echo "hf_energy_frac >= ${HF_THRESH}"
echo "top-percentile:   ${TOP_PERCENTILE:-<unused>}"
echo "Extra args:       ${EXTRA_ARGS}"
echo "Start:            $(date)"
echo "================"

cd "${PROJECT_DIR}"

TOP_ARG=()
if [[ -n "${TOP_PERCENTILE}" ]]; then
    TOP_ARG=(--top-percentile "${TOP_PERCENTILE}")
fi

# shellcheck disable=SC2086
python "${PROJECT_DIR}/scripts/score_image_noise.py" \
    --patch-root       "${PATCH_ROOT}" \
    --out-csv          "${OUT_CSV}" \
    --out-json         "${OUT_JSON}" \
    --exclude-patterns ${EXCLUDE_PATTERNS} \
    --hp-thresh        "${HP_THRESH}" \
    --hf-thresh        "${HF_THRESH}" \
    "${TOP_ARG[@]}" \
    ${EXTRA_ARGS} \
    2>&1

EXIT_CODE=$?

echo "=== Done ==="
echo "Exit code: ${EXIT_CODE}"
echo "End:       $(date)"

exit ${EXIT_CODE}
