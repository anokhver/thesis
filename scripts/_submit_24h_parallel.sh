#!/usr/bin/env bash
# Parallel version of _run_24h_full.sh for MetaCentrum (PBS): submits ONE
# job per session so all sessions run at the same time, plus a merge job
# that starts automatically once every session job has finished OK and
# writes outs/puncta_morphology.csv + outs/puncta_morphology_images.csv.
#
# Run it on the login node (it only submits jobs):
#   bash scripts/_submit_24h_parallel.sh                     # all 4 sessions
#   bash scripts/_submit_24h_parallel.sh 20251219 20251220   # only these; the
#                                                            # merge still uses all 4
# Settings can be overridden: NCPUS=16 WALLTIME=6:00:00 bash scripts/_submit_24h_parallel.sh
# Dry run (print the qsub commands, submit nothing): QSUB=echo bash scripts/_submit_24h_parallel.sh
set -euo pipefail

REPO=/storage/brno2/home/anokhver/thesis
PY=${PY:-/storage/brno2/home/anokhver/.conda/envs/microscopy/bin/python}
QSUB=${QSUB:-qsub}
NCPUS=${NCPUS:-32}
MEM=${MEM:-64gb}
WALLTIME=${WALLTIME:-12:00:00}

ALL_SESSIONS=(20251205 20251213 20251219 20251220)
if [ "$#" -gt 0 ]; then SESSIONS=("$@"); else SESSIONS=("${ALL_SESSIONS[@]}"); fi

OUT_DIR=outs/puncta_morphology_24h
WIDE_CFG=configs/puncta/wide_range.json
cd "$REPO"
mkdir -p "$OUT_DIR"

job_ids=()
for d in "${SESSIONS[@]}"; do
    cmd="cd $REPO && export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
$PY scripts/analysis/puncta_morphology.py \
--input_patches data/patches_128_from_zip/$d \
--time_filter '24[_ ]?hod' \
--dendrite_dir data/pseudolabels/dend/$d \
--output_csv $OUT_DIR/$d.csv \
--group_patterns configs/clustering/group_patterns.json \
--cfg_pre $WIDE_CFG --cfg_post $WIDE_CFG \
--workers \$NCPUS > $OUT_DIR/$d.log 2>&1"
    id=$($QSUB -m a -N "puncta_$d" -l "select=1:ncpus=$NCPUS:mem=$MEM" \
        -l "walltime=$WALLTIME" -- /bin/bash -c "$cmd")
    echo "session $d -> job $id (log: $OUT_DIR/$d.log)"
    job_ids+=("$id")
done

merge_py="
import pandas as pd
from pathlib import Path
out_dir = Path('$OUT_DIR')
sessions = '${ALL_SESSIONS[*]}'.split()
for suffix, target in (('', 'outs/puncta_morphology.csv'), ('_images', 'outs/puncta_morphology_images.csv')):
    frames = [pd.read_csv(out_dir / f'{d}{suffix}.csv', sep=';') for d in sessions]
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(target, sep=';', index=False, encoding='utf-8')
    print('wrote', target, len(merged), 'rows')
"
depend=$(IFS=:; echo "${job_ids[*]}")
merge_id=$($QSUB -m ae -N puncta_merge -W "depend=afterok:$depend" \
    -l select=1:ncpus=1:mem=8gb -l walltime=0:30:00 -- /bin/bash -c \
    "cd $REPO && $PY -c \"$merge_py\" > outs/puncta_merge.log 2>&1")
echo "merge -> job $merge_id (starts when all session jobs finish OK; emails you; log: outs/puncta_merge.log)"
