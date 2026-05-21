#!/bin/bash
#PBS -N cluster_py
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_mem=24gb:scratch_local=50gb
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/
#PBS -m abe
#PBS -M veronika.i.anokhina@gmail.com

# End-to-end clustering + group-statistics run.
#
# Two modes (driven by the variables below):
#   A. CHECKPOINT=...  -> run extract_full_embeddings.py first, then cluster.
#                         Requires GPU resources (default in the #PBS lines).
#   B. BUNDLE_DIR=...  -> cluster an already-extracted bundle directly.
#                         GPU is unused; you may submit with a CPU-only line.
#
# Submit with `qsub`; override defaults via -v, e.g.:
#   qsub -v CHECKPOINT=/path/to/best_model.pt scripts/metacentrum/submit_clustering_py.sh
#   qsub -v BUNDLE_DIR=/path/to/data/embeddings/<run> scripts/metacentrum/submit_clustering_py.sh

set -euo pipefail

PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
CONDA_ENV="microscopy"

# ----------------------------------------------------------------------
# >>> EDIT BEFORE SUBMITTING (or pass via `qsub -v ...`) <<<
# ----------------------------------------------------------------------
CHECKPOINT="${CHECKPOINT:-}"               # .pt file or model run directory
BUNDLE_DIR="${BUNDLE_DIR:-}"               # alternative: pre-extracted bundle
OUTPUT_DIR="${OUTPUT_DIR:-}"               # default: <bundle>/clustering
OUTPUT_ROOT="${OUTPUT_ROOT:-}"             # default: data/embeddings (extraction)
RUN_NAME="${RUN_NAME:-}"                   # override bundle folder name
DATA_ROOT="${DATA_ROOT:-}"                 # default: data/patches_128_from_zip
CONTROL_PATTERN="${CONTROL_PATTERN:-}"     # e.g. DMSO; empty = disabled
EXCLUDE_DATES="${EXCLUDE_DATES:-}"         # space-separated date folders to skip
# Default to the clustering-only exclude list (keeps every KONTROLA / control
# tag so the bundle has a real negative-control group). Use
# data/data_analysis/exclude.json instead to mirror SSL pretraining (drops
# KONTROLA + protocol variants). Set EXCLUDE_PATTERNS_FILE='' to disable.
EXCLUDE_PATTERNS_FILE="${EXCLUDE_PATTERNS_FILE:-${PROJECT_DIR}/data/data_analysis/exclude_clustering.json}"
EXTRACT_OVERWRITE="${EXTRACT_OVERWRITE:-}" # set to 1 to re-extract
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-128}"
EXTRACT_NUM_WORKERS="${EXTRACT_NUM_WORKERS:-4}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"             # any extra run_clustering.py flags

mkdir -p "${PROJECT_DIR}/logs"

source /cvmfs/software.metacentrum.cz/conda/envs/miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate /storage/brno2/home/anokhver/.conda/envs/${CONDA_ENV}

# Conservative BLAS thread caps; the pipeline parallelises elsewhere.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

cd "${PROJECT_DIR}"

echo "=== Job Info ==="
echo "Job ID:     ${PBS_JOBID:-<interactive>}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Conda env:  ${CONDA_ENV}"
echo "Project:    ${PROJECT_DIR}"
echo "Checkpoint: ${CHECKPOINT:-<none>}"
echo "Bundle:     ${BUNDLE_DIR:-<derive from checkpoint>}"
echo "Output:     ${OUTPUT_DIR:-<bundle>/clustering}"
echo "Control:    ${CONTROL_PATTERN:-<none>}"
echo "Exclude:    ${EXCLUDE_DATES:-<none>}"
echo "ExcludePat: ${EXCLUDE_PATTERNS_FILE:-<none>}"
echo "Extra:      ${EXTRA_FLAGS:-<none>}"
echo "Start:      $(date)"
echo "================"

if [[ -z "${CHECKPOINT}" && -z "${BUNDLE_DIR}" ]]; then
    echo "ERROR: set CHECKPOINT=... or BUNDLE_DIR=... via -v" >&2
    exit 2
fi
if [[ -n "${CHECKPOINT}" && -n "${BUNDLE_DIR}" ]]; then
    echo "ERROR: set only one of CHECKPOINT / BUNDLE_DIR" >&2
    exit 2
fi

CMD=( python "${PROJECT_DIR}/scripts/run_clustering.py" )

if [[ -n "${CHECKPOINT}" ]]; then
    if [[ ! -e "${CHECKPOINT}" ]]; then
        echo "ERROR: CHECKPOINT does not exist: ${CHECKPOINT}" >&2
        exit 2
    fi
    CMD+=( --checkpoint "${CHECKPOINT}" )
    [[ -n "${OUTPUT_ROOT}" ]] && CMD+=( --output-root "${OUTPUT_ROOT}" )
    [[ -n "${RUN_NAME}"    ]] && CMD+=( --run-name "${RUN_NAME}" )
    [[ -n "${DATA_ROOT}"   ]] && CMD+=( --data-root "${DATA_ROOT}" )
    CMD+=( --extract-batch-size "${EXTRACT_BATCH_SIZE}" )
    CMD+=( --extract-num-workers "${EXTRACT_NUM_WORKERS}" )
    if [[ -n "${EXCLUDE_DATES}" ]]; then
        # shellcheck disable=SC2206
        EXCL_ARR=( ${EXCLUDE_DATES} )
        CMD+=( --exclude-dates "${EXCL_ARR[@]}" )
    fi
    if [[ -n "${EXCLUDE_PATTERNS_FILE}" ]]; then
        if [[ ! -f "${EXCLUDE_PATTERNS_FILE}" ]]; then
            echo "ERROR: EXCLUDE_PATTERNS_FILE not found: ${EXCLUDE_PATTERNS_FILE}" >&2
            exit 2
        fi
        CMD+=( --exclude-patterns-file "${EXCLUDE_PATTERNS_FILE}" )
    fi
    [[ -n "${EXTRACT_OVERWRITE}" ]] && CMD+=( --extract-overwrite )
else
    if [[ ! -d "${BUNDLE_DIR}" ]]; then
        echo "ERROR: BUNDLE_DIR does not exist: ${BUNDLE_DIR}" >&2
        exit 2
    fi
    CMD+=( --bundle "${BUNDLE_DIR}" )
fi

[[ -n "${OUTPUT_DIR}"      ]] && CMD+=( --output-dir "${OUTPUT_DIR}" )
[[ -n "${CONTROL_PATTERN}" ]] && CMD+=( --control-pattern "${CONTROL_PATTERN}" )

# shellcheck disable=SC2206
EXTRA_ARR=( ${EXTRA_FLAGS} )
if (( ${#EXTRA_ARR[@]} > 0 )); then
    CMD+=( "${EXTRA_ARR[@]}" )
fi

echo "Running: ${CMD[*]}"
set +e
"${CMD[@]}" 2>&1
EXIT_CODE=$?
set -e

echo "=== Done ==="
echo "Exit code: ${EXIT_CODE}"
echo "End:       $(date)"
exit ${EXIT_CODE}
