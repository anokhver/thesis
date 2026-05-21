#!/bin/bash
#PBS -N extract_masks
#PBS -l select=1:ncpus=4:mem=4gb:scratch_local=20gb
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -o /storage/brno2/home/anokhver/thesis/logs/

# One-shot extraction of mask tarballs to data/pseudolabels/.
# Each tarball is ~hundreds of k tiny .npy files; the bottleneck is
# metadata writes on the NFS, not gzip. 6h walltime covers worst case.
#
# Select which archives to extract via MASKS env (space-separated keys):
#   qsub scripts/metacentrum/submit_extract_masks.sh                # all
#   qsub -v MASKS="puncta"           scripts/metacentrum/submit_extract_masks.sh
#   qsub -v MASKS="soma dend"        scripts/metacentrum/submit_extract_masks.sh
#
# Each key resolves to a tarball + destination dir below.

PROJECT_DIR="/storage/brno2/home/anokhver/thesis"
set -euo pipefail
cd "${PROJECT_DIR}"

# Map: key -> tarball filename (relative to data/pseudolabels/).
# Destination dir is always ``data/pseudolabels/<key>``.
declare -A TARBALLS=(
    [soma]=soma.tar.gz
    [dend]=dend.tar.gz
    [puncta]=log-puncta.tar.gz
)

MASKS="${MASKS:-soma dend puncta}"

echo "start: $(date)"
echo "node:  $(hostname)"
echo "masks: ${MASKS}"

mkdir -p data/pseudolabels
for arc in ${MASKS}; do
    if [[ -z "${TARBALLS[$arc]:-}" ]]; then
        echo "ERROR: unknown mask key '${arc}' (known: ${!TARBALLS[*]})" >&2
        exit 2
    fi
    tarball="data/pseudolabels/${TARBALLS[$arc]}"
    dest="data/pseudolabels/${arc}"
    if [[ ! -f "${tarball}" ]]; then
        echo "ERROR: tarball ${tarball} not found" >&2
        exit 2
    fi
    expected_count=$(tar tzf "${tarball}" | grep -c '\.npy$' || echo 0)

    if [[ -d "${dest}" ]]; then
        present=$(find "${dest}" -type f -name '*.npy' | wc -l)
        if [[ "${present}" -eq "${expected_count}" ]]; then
            echo "skip ${arc}: already fully extracted (${present}/${expected_count})"
            continue
        fi
        echo "${arc}: partial extraction (${present}/${expected_count}); removing and re-extracting"
        rm -rf "${dest}"
    fi
    echo "extracting ${tarball} -> ${dest}"
    t0=$(date +%s)
    tar xzf "${tarball}" -C data/pseudolabels/
    t1=$(date +%s)
    files=$(find "${dest}" -type f -name '*.npy' | wc -l)
    echo "  done in $((t1-t0))s: ${files}/${expected_count} files, $(du -sh ${dest} | cut -f1)"
done

echo "end:   $(date)"


