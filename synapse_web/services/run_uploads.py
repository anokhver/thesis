"""Run-input ingestion: safe ZIP extraction + manifest inspection.

Two input shapes are supported:

* **Patches** (for ``extract_and_cluster`` runs): a directory in the
  ``synaptic_ssl.utils_data.PatchDataset`` format — an ``index.csv``
  next to ``.npy`` files. PatchDataset also tolerates a single
  nested layout (``<sub>/index.csv``); we mirror that.
* **Bundle** (for ``cluster_only`` runs): the directory passed to
  ``scripts/run_clustering.py --bundle`` — ``embeddings.npy`` plus
  ``metadata.csv`` (and an optional ``manifest.json``). ``run_clustering``
  refuses to start if ``metadata.csv`` lacks a ``source_image`` column,
  so we enforce the same here at upload time.

Security: ZIP extraction streams every member through size and path
guards and never calls ``ZipFile.extract``/``extractall``. See
``safe_extract_zip`` for the full guard list.
"""
from __future__ import annotations

import csv
import io
import logging
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable

from django.conf import settings
from django.core.exceptions import ValidationError

from .naming import derive_treatment_group

logger = logging.getLogger(__name__)

_CHUNK = 64 * 1024
_BAD_NAME_CHARS = re.compile(r"[:\x00]")
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except (ValueError, OSError):
        return False
    return True


def _zip_entry_is_symlink(info: zipfile.ZipInfo) -> bool:
    """Unix mode 0o120000 marks a symlink in the upper bits of external_attr."""
    return (info.external_attr >> 16) & 0o170000 == 0o120000


def _validate_member_name(name: str) -> PurePosixPath:
    """Return a sanitised relative posix path, or raise ValidationError."""
    if not name:
        raise ValidationError("ZIP contains an unnamed entry.")
    if "\\" in name:
        raise ValidationError(f"ZIP entry uses backslash: {name!r}.")
    if _BAD_NAME_CHARS.search(name):
        raise ValidationError(f"ZIP entry has unsafe characters: {name!r}.")
    if _DRIVE_PREFIX.match(name):
        raise ValidationError(f"ZIP entry uses a Windows drive prefix: {name!r}.")
    rel = PurePosixPath(name)
    if rel.is_absolute() or str(rel).startswith("/"):
        raise ValidationError(f"ZIP entry is absolute: {name!r}.")
    parts = rel.parts
    if any(p == ".." for p in parts):
        raise ValidationError(f"ZIP entry escapes its root: {name!r}.")
    return rel


def _is_mac_metadata(rel: PurePosixPath) -> bool:
    return rel.parts and (rel.parts[0] == "__MACOSX" or rel.name == ".DS_Store")


def _strip_common_prefix(rels: list[PurePosixPath]) -> list[PurePosixPath]:
    """If every entry sits under a single shared top-level directory, drop it."""
    if not rels:
        return rels
    if not all(len(r.parts) > 1 for r in rels):
        return rels
    first = rels[0].parts[0]
    if not all(r.parts[0] == first for r in rels):
        return rels
    return [PurePosixPath(*r.parts[1:]) for r in rels]


def safe_extract_zip(file_obj, dest_dir: Path) -> None:
    """Extract a user-uploaded ZIP into dest_dir with strict guards.

    Guards:
    - Reject if not a valid zip, encrypted, or larger than configured
      compressed/uncompressed/member-count limits.
    - Reject path traversal, absolute paths, backslashes, colons,
      Windows drive prefixes, ``__MACOSX/``-style metadata, and
      symlink entries.
    - Reject duplicate destinations under case-insensitive comparison
      (so ``A.npy`` and ``a.npy`` cannot collide on Windows).
    - Strip a single common top-level directory so users can zip
      either the folder or its contents.

    On any failure, the caller is responsible for cleaning ``dest_dir``.
    """
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    max_members = int(settings.MAX_ZIP_MEMBERS)
    max_total = int(settings.MAX_ZIP_UNCOMPRESSED_BYTES)

    try:
        zf = zipfile.ZipFile(file_obj)
    except zipfile.BadZipFile as exc:
        raise ValidationError(f"Invalid ZIP file: {exc}.")

    with zf:
        infos = zf.infolist()
        if len(infos) > max_members:
            raise ValidationError(
                f"ZIP has too many entries ({len(infos)} > {max_members})."
            )

        # First pass: validate every entry, refuse the whole archive on any
        # red flag, and compute the destination relative paths.
        validated: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        total_uncompressed = 0
        for info in infos:
            if info.flag_bits & 0x1:
                raise ValidationError("ZIP contains encrypted entries.")
            rel = _validate_member_name(info.filename)
            if _is_mac_metadata(rel):
                continue
            if info.is_dir():
                continue
            if _zip_entry_is_symlink(info):
                raise ValidationError(
                    f"ZIP entry is a symlink: {info.filename!r}."
                )
            total_uncompressed += int(info.file_size)
            if total_uncompressed > max_total:
                raise ValidationError(
                    f"ZIP is too large uncompressed "
                    f"({total_uncompressed} > {max_total} bytes)."
                )
            validated.append((info, rel))

        if not validated:
            raise ValidationError("ZIP is empty.")

        rels = _strip_common_prefix([rel for _, rel in validated])
        validated = list(zip([info for info, _ in validated], rels))

        # Case-insensitive collision check across the whole archive.
        seen: dict[str, str] = {}
        for _, rel in validated:
            key = str(rel).casefold()
            if key in seen and seen[key] != str(rel):
                raise ValidationError(
                    f"ZIP has case-colliding entries: {seen[key]!r} vs {rel!r}."
                )
            seen[key] = str(rel)

        # Second pass: stream the bytes, enforce a running budget, refuse
        # any path that escapes dest_dir after resolution.
        bytes_written = 0
        for info, rel in validated:
            if not rel.parts:
                continue
            target = (dest_dir / Path(*rel.parts)).resolve()
            if not _is_within(target, dest_dir):
                raise ValidationError(
                    f"ZIP entry escapes destination: {info.filename!r}."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                while True:
                    chunk = src.read(_CHUNK)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > max_total:
                        raise ValidationError(
                            "ZIP exceeds uncompressed-byte budget."
                        )
                    dst.write(chunk)


def _find_single_subdir_with(dest_dir: Path, required: Iterable[str]) -> Path | None:
    """Return dest_dir, or its sole subdir, if all `required` files exist there."""
    if all((dest_dir / name).is_file() for name in required):
        return dest_dir
    subdirs = [p for p in dest_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and all((subdirs[0] / name).is_file() for name in required):
        return subdirs[0]
    return None


def _index_csv_paths(root: Path) -> list[Path]:
    flat = root / "index.csv"
    if flat.is_file():
        return [flat]
    return sorted(p for p in root.glob("*/index.csv") if p.is_file())


def _read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def inspect_patches(dest_dir: Path) -> list[dict]:
    """Build a per-source-image manifest from an extracted PatchDataset."""
    indexes = _index_csv_paths(dest_dir)
    if not indexes:
        raise ValidationError(
            "Patches upload must contain an index.csv at the root or in a "
            "single nested subdirectory."
        )

    counter: Counter[str] = Counter()
    for idx in indexes:
        rows = _read_csv_rows(idx)
        if not rows:
            continue
        if "source_image" not in rows[0]:
            raise ValidationError(
                f"index.csv at {idx.relative_to(dest_dir)} is missing the "
                f"required 'source_image' column."
            )
        for row in rows:
            src = (row.get("source_image") or "").strip()
            if src:
                counter[src] += 1

    if not counter:
        raise ValidationError("Patches upload contains no rows with a source_image.")

    return [
        {
            "source_image": name,
            "treatment_group": derive_treatment_group(name),
            "n_patches": n,
        }
        for name, n in counter.most_common()
    ]


def inspect_bundle(dest_dir: Path) -> tuple[list[dict], Path]:
    """Build manifest + return the actual bundle root from an extracted bundle."""
    bundle_root = _find_single_subdir_with(
        dest_dir, ("embeddings.npy", "metadata.csv")
    )
    if bundle_root is None:
        raise ValidationError(
            "Bundle upload must contain embeddings.npy and metadata.csv "
            "(at the root or in a single nested subdirectory)."
        )

    rows = _read_csv_rows(bundle_root / "metadata.csv")
    if not rows:
        raise ValidationError("metadata.csv is empty.")
    if "source_image" not in rows[0]:
        raise ValidationError(
            "metadata.csv is missing the required 'source_image' column."
        )

    counter: Counter[str] = Counter()
    for row in rows:
        src = (row.get("source_image") or "").strip()
        if src:
            counter[src] += 1
    if not counter:
        raise ValidationError("metadata.csv has no rows with a source_image.")

    manifest = [
        {
            "source_image": name,
            "treatment_group": derive_treatment_group(name),
            "n_patches": n,
        }
        for name, n in counter.most_common()
    ]
    return manifest, bundle_root


def remove_tree(path: Path) -> None:
    """Best-effort recursive delete, swallowing errors."""
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        logger.warning("Could not clean up %s", path)
