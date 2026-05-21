#!/usr/bin/env python3
r"""Gather full metadata for a pseudolabel damage-detection experiment.

Reads the experiment summary JSON (``patch_root``, ``flagged_sources``,
``exclude_patterns``, ...) and emits a full nested report plus two flat
CSVs (flagged sources, all sources). See
:mod:`synaptic_ssl.utils_metadata.experiment_report` for the
report-building logic, and :mod:`synaptic_ssl.utils_metadata.olympus_matcher`
for the optional ``.oex``/``.vsi`` enrichment triggered by
``--metadata_root``.

Outputs
-------
``<output_dir>/experiment_metadata_full.json``
    Full nested report with config, derived stats, and all per-source data.
``<output_dir>/flagged_sources_metadata.csv``
    Flat table for only flagged sources.
``<output_dir>/all_sources_metadata.csv``
    Flat table for every source found in the patch index.

Usage::

    python scripts/metadata/gather_experiment_metadata.py \
        --experiment_json /path/to/experiment_summary.json \
        --output_dir /path/to/output

    python scripts/metadata/gather_experiment_metadata.py \
        --experiment_json_inline '{"patch_root": "...", "flagged_sources": [...]}' \
        --output_dir /path/to/output
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from synaptic_ssl.utils_metadata.experiment_report import (
    build_report, load_experiment_json, write_outputs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gather full metadata for a specific experiment JSON (including flagged sources)."
    )
    parser.add_argument("--experiment_json", type=Path, default=None,
                        help="Path to JSON file with experiment summary.")
    parser.add_argument("--experiment_json_inline", type=str, default=None,
                        help="Inline JSON string with experiment summary.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory where metadata outputs will be written.")
    parser.add_argument("--metadata_root", action="append", type=Path, default=[],
                        help=("Root directory to recursively scan for Olympus "
                              ".oex/.vsi metadata files. Repeat to provide multiple roots."))
    args = parser.parse_args()

    experiment = load_experiment_json(args.experiment_json, args.experiment_json_inline)
    report = build_report(experiment, metadata_roots=list(args.metadata_root or []))

    full_json, flagged_csv, all_csv = write_outputs(report, args.output_dir)

    logger.info(f"Wrote full report: {full_json}")
    logger.info(f"Wrote flagged CSV: {flagged_csv}")
    logger.info(f"Wrote all-sources CSV: {all_csv}")

    derived = report["derived"]
    logger.info(
        "Summary: index sources=%d, flagged(input)=%d, flagged(in index)=%d, missing=%d",
        derived["index_sources_count"],
        derived["flagged_sources_count_input"],
        derived["flagged_sources_in_index"],
        len(derived["flagged_sources_missing_from_index"]),
    )


if __name__ == "__main__":
    main()
