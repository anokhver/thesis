"""IO helpers for joint 2-channel (PRE/POST) Spotiflow pseudolabels.

The on-disk format produced by ``scripts/pseudolabels/puncta_spotiflow_from_mip.py``
in *MIP mode* (the layout we have in ``data/pseudolabels/spotiflow.tar.gz``)
is::

    spotiflow/
        <session>/
            <source_stem>_pre.npy    # (H, W) uint8, full reassembled MIP
            <source_stem>_post.npy   # (H, W) uint8

Random access on a ``.tar.gz`` is infeasible (the gzip stream has no
seek points), and ~9 GB of uint8 masks does not comfortably fit in RAM.
The pragmatic compromise is a **one-time idempotent extraction** to a
cache directory plus ``np.load(mmap_mode='r')`` for per-tile slicing.
The cache dir can be deleted at any time and is rebuilt on next run.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from typing import Iterable


def extract_pseudolabel_archive(
    archive_path: str | Path,
    cache_dir: str | Path,
    *,
    sessions: Iterable[str] | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    """Extract ``spotiflow.tar.gz`` to ``cache_dir`` once (idempotent).

    Parameters
    ----------
    archive_path
        Path to ``spotiflow.tar.gz``. Top-level entry must be ``spotiflow/``.
    cache_dir
        Destination directory. The archive is extracted *into* this dir
        so the final layout is ``cache_dir/spotiflow/<session>/...``.
    sessions
        Optional iterable of session names (e.g. ``["20251219"]``). When
        set, only members under ``spotiflow/<session>/`` are extracted.
    logger
        Optional logger for progress messages.

    Returns
    -------
    pathlib.Path
        Path to ``cache_dir/spotiflow/`` (the dir containing one
        sub-directory per session).
    """
    log = (logger.info if logger is not None else print)
    archive_path = Path(archive_path)
    cache_dir = Path(cache_dir)
    target_root = cache_dir / "spotiflow"

    if sessions is not None:
        sessions = set(sessions)

    # Idempotent skip: only consider extraction complete if every requested
    # session directory exists. Partial extractions are *not* skipped.
    if target_root.is_dir():
        present = {p.name for p in target_root.iterdir() if p.is_dir()}
        if sessions is None or sessions.issubset(present):
            log(f"pseudolabel cache already present at {target_root}")
            return target_root

    cache_dir.mkdir(parents=True, exist_ok=True)
    log(f"extracting {archive_path} -> {cache_dir}")
    if not archive_path.exists():
        raise FileNotFoundError(f"pseudolabel archive not found: {archive_path}")

    n_extracted = 0
    with tarfile.open(archive_path, mode="r:gz") as tar:
        for member in tar:
            if not member.isfile():
                # still extract directory entries so paths exist
                tar.extract(member, path=cache_dir)
                continue
            if sessions is not None:
                parts = Path(member.name).parts
                # expected: ("spotiflow", "<session>", "<file>.npy")
                if len(parts) < 3 or parts[1] not in sessions:
                    continue
            tar.extract(member, path=cache_dir)
            n_extracted += 1
    log(f"extracted {n_extracted} files to {target_root}")
    return target_root


def discover_full_image_masks(
    mask_root: str | Path,
    *,
    sessions: Iterable[str] | None = None,
    pre_suffix: str = "_pre.npy",
    post_suffix: str = "_post.npy",
) -> dict[str, dict[str, Path]]:
    """Index full-image PRE+POST mask paths under ``mask_root``.

    Returns a dict keyed by ``"<session>/<source_stem>"`` with values
    ``{"pre": Path, "post": Path}``. Sources missing either channel are
    skipped with a warning.
    """
    mask_root = Path(mask_root)
    if sessions is not None:
        session_dirs = [mask_root / s for s in sessions]
    else:
        session_dirs = [p for p in mask_root.iterdir() if p.is_dir()]

    index: dict[str, dict[str, Path]] = {}
    for sess_dir in session_dirs:
        if not sess_dir.is_dir():
            continue
        sess = sess_dir.name
        # Build stem -> {pre,post} for this session
        pre_files = {p.name[: -len(pre_suffix)]: p for p in sess_dir.glob(f"*{pre_suffix}")}
        post_files = {p.name[: -len(post_suffix)]: p for p in sess_dir.glob(f"*{post_suffix}")}
        common = set(pre_files) & set(post_files)
        for stem in common:
            index[f"{sess}/{stem}"] = {
                "pre": pre_files[stem],
                "post": post_files[stem],
            }
    return index


__all__ = [
    "extract_pseudolabel_archive",
    "discover_full_image_masks",
]
