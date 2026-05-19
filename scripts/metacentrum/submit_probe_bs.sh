#!/bin/bash
#PBS -N probe_bs
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_mem=80gb:scratch_local=20gb
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/

# Probe the largest batch size that fits in VRAM for each of the three
# 128_* config variants (default, vicreg_off, vicreg_on). Writes one log
# section per variant to the standard PBS .OU file.

PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
CONDA_ENV="microscopy"

set -euo pipefail

mkdir -p "${PROJECT_DIR}/logs"

source /cvmfs/software.metacentrum.cz/conda/envs/miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate /storage/brno2/home/anokhver/.conda/envs/${CONDA_ENV}

cd "${PROJECT_DIR}"

echo "=== Job Info ==="
echo "Job ID:    ${PBS_JOBID:-<none>}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "VRAM:      $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Conda env: ${CONDA_ENV}"
echo "Start:     $(date)"
echo "================"
echo

# moby has identical model shape to tinny22k, so probing the moby trio
# covers both. Scratch has the same shape too. The only axis that changes
# memory is ssl.w_vicreg (0 -> 2 fwd, >0 -> 4 fwd).
CONFIGS=(
    "configs/pretrain_moby/128_default.json"
    "configs/pretrain_moby/128_no_fourier_vicreg_off.json"
    "configs/pretrain_moby/128_no_fourier_vicreg_on.json"
)

for CFG in "${CONFIGS[@]}"; do
    echo
    echo "######################################################################"
    echo "# probing: ${CFG}"
    echo "######################################################################"
    python scripts/probe_batch_size.py \
        --config "${CFG}" \
        --sizes 32,48,64,96,128,160,192,224,256,320,384 \
        --steps 2 \
        2>&1 || echo "[warn] probe exited with non-zero status, continuing"
done

echo
echo "=== Done ==="
echo "End: $(date)"
