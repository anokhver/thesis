#!/bin/bash
#PBS -N extract_masks
#PBS -l select=1:ncpus=2:mem=4gb:scratch_local=20gb
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/

# One-shot extraction of soma+dend mask tarballs to data/pseudolabels/
# (login node is throttled; a worker node finishes this in a few minutes).

PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
set -euo pipefail
cd "${PROJECT_DIR}"

echo "start: $(date)"
echo "node:  $(hostname)"

mkdir -p data/pseudolabels
for arc in soma dend; do
    if [[ -d "data/pseudolabels/${arc}" ]]; then
        echo "skip ${arc}: already extracted"
        continue
    fi
    echo "extracting ${arc}.tar.gz -> data/pseudolabels/${arc}"
    tar xzf "data/pseudolabels/${arc}.tar.gz" -C data/pseudolabels/
    echo "  done: $(du -sh data/pseudolabels/${arc})  files=$(find data/pseudolabels/${arc} -type f | wc -l)"
done

echo "end:   $(date)"
