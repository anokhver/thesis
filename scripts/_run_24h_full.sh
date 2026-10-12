#!/usr/bin/env bash
# Orchestration helper for the "24h" full run across all matching sessions.
# Not part of the CLI deliverable itself -- just chains per-session
# scripts/analysis/puncta_morphology.py invocations (patches mode) and
# concatenates the results. Safe to delete once the run is done.
set -euo pipefail
cd /storage/brno2/home/anokhver/thesis

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

SESSIONS=(20251205 20251213 20251219 20251220)
OUT_DIR=outs/puncta_morphology_24h
mkdir -p "$OUT_DIR"

for d in "${SESSIONS[@]}"; do
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] session $d ==="
    conda run -n microscopy python scripts/analysis/puncta_morphology.py \
        --input_patches "data/patches_128_from_zip/$d" \
        --time_filter "24[_ ]?hod" \
        --dendrite_dir "data/pseudolabels/dend/$d" \
        --output_csv "$OUT_DIR/$d.csv" \
        --group_patterns configs/clustering/group_patterns.json \
        --workers 8
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] session $d done ==="
done

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] concatenating ==="
conda run -n microscopy python -c "
import pandas as pd
from pathlib import Path

out_dir = Path('outs/puncta_morphology_24h')
sessions = ['20251205', '20251213', '20251219', '20251220']

main = pd.concat([pd.read_csv(out_dir / f'{d}.csv', sep=';') for d in sessions], ignore_index=True)
main.to_csv('outs/puncta_morphology.csv', sep=';', index=False, encoding='utf-8')
print('wrote outs/puncta_morphology.csv', len(main), 'rows')

images = pd.concat([pd.read_csv(out_dir / f'{d}_images.csv', sep=';') for d in sessions], ignore_index=True)
images.to_csv('outs/puncta_morphology_images.csv', sep=';', index=False, encoding='utf-8')
print('wrote outs/puncta_morphology_images.csv', len(images), 'rows')
"
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] ALL DONE ==="
