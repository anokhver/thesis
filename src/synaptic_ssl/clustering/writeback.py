"""Append cluster-label columns into the patch-directory ``index.csv``.

Idempotent and atomic: new columns are appended; existing columns are
overwritten. Unmatched filenames receive ``-2`` so downstream code can
distinguish "noise" (``-1``) from "not in this run" (``-2``).
"""
from __future__ import annotations

import numpy as np


def write_cluster_columns(
    data_root, filenames: np.ndarray, column_to_labels: dict,
    *, dry_run: bool = False,
) -> dict:
    """Atomic + idempotent. New columns appended; existing overwritten."""
    import os
    from pathlib import Path
    import pandas as pd

    csv_path = Path(data_root) / "index.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise KeyError("expected a 'filename' column in index.csv")
    original_cols = list(df.columns)

    summary = {}
    for col, labels in column_to_labels.items():
        s = pd.Series(np.asarray(labels, dtype=np.int64), index=filenames, name=col)
        aligned = df["filename"].map(s).fillna(-2).astype(np.int64)
        action = "overwrote" if col in df.columns else "appended"
        df[col] = aligned.values
        summary[col] = {
            "action": action,
            "n_assigned": int((aligned != -2).sum()),
            "n_unmatched": int((aligned == -2).sum()),
            "n_clusters": int(len(set(int(v) for v in labels if v != -1))),
        }

    new_cols = [c for c in df.columns if c not in original_cols]
    df = df[[c for c in original_cols if c in df.columns] + new_cols]

    if dry_run:
        return summary
    tmp = csv_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)
    return summary
