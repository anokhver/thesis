"""Microscopy metadata extraction, source-name parsing, and experiment reports.

Submodules:
  * :mod:`.source_names`  — parse dataset filenames into experiment fields.
  * :mod:`.extractors`    — per-format raw metadata extractors (lazy on
                            ``aicsimageio`` / ``tifffile``).
  * :mod:`.flatten`       — flatten raw metadata into one CSV row.
  * :mod:`.sessions`      — scan a folder tree, process one session per
                            subfolder.
  * :mod:`.olympus_matcher` — match source basenames to ``.oex`` / ``.vsi``
                              files for enrichment.
  * :mod:`.experiment_report` — assemble a full per-experiment report.
  * :mod:`.z_range`       — per-image z-bounds scan (lazy ``aicsimageio``).

Most heavy readers (``aicsimageio``, ``tifffile``, ``ome_types``) are
imported lazily inside functions so ``import synaptic_ssl.utils_metadata``
works without the optional ``imaging`` extra installed.
"""

from .source_names import (
    parse_source_name,
    source_match_keys,
    path_match_keys,
    basename,
    source_stem,
)

__all__ = [
    "parse_source_name",
    "source_match_keys",
    "path_match_keys",
    "basename",
    "source_stem",
]
