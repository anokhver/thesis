"""Checkpoint registry — discover, upload, and delete encoder weights.

The runtime store is `settings.CHECKPOINT_DIR`. Files are kept flat
(no subdirectories) so they can be referenced from `AnalysisRun`
by a single relative name.

Security note: this is a localhost developer tool with no auth.
`.pt` files are pickle-backed; whoever can POST to /checkpoints/
can later have Phase E load them. Phase E should use
`torch.load(..., weights_only=True)` where possible. Add auth if
this app is ever exposed beyond localhost.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from django.conf import settings
from django.core.exceptions import ValidationError
from django.utils.text import slugify

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".pt", ".pth"}
_PARTIAL_SUFFIX = ".partial"


def _root() -> Path:
    return Path(settings.CHECKPOINT_DIR)


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except (ValueError, OSError):
        return False
    return True


def _human_size(n: int) -> str:
    step = 1024.0
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(n)
    for unit in units:
        if size < step or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= step
    return f"{n} B"


def list_checkpoints() -> list[dict]:
    """Return checkpoint metadata sorted by mtime desc.

    Skips: missing dir, hidden files (`.`-prefixed), symlinks,
    in-flight `.partial` writes, anything whose suffix is not
    in `ALLOWED_SUFFIXES` (case-insensitive).
    """
    root = _root()
    if not root.exists():
        return []

    entries: list[dict] = []
    try:
        children: Iterable[Path] = root.iterdir()
    except OSError:
        return []

    for entry in children:
        name = entry.name
        if name.startswith("."):
            continue
        if name.endswith(_PARTIAL_SUFFIX):
            continue
        try:
            if entry.is_symlink() or not entry.is_file():
                continue
        except OSError:
            continue
        if entry.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        entries.append(
            {
                "name": name,
                "relative_path": name,
                "size_bytes": stat.st_size,
                "size_display": _human_size(stat.st_size),
                "modified_at": datetime.fromtimestamp(stat.st_mtime),
            }
        )

    entries.sort(key=lambda item: item["modified_at"], reverse=True)
    return entries


def validate_checkpoint_upload(uploaded_file) -> str:
    """Return the safe filename to write. Pure — no filesystem writes.

    Raises `ValidationError` on any policy violation. Filesystem
    collision check is best-effort (race-free check happens at the
    atomic-rename step).
    """
    raw_name = getattr(uploaded_file, "name", "") or ""
    if "/" in raw_name or "\\" in raw_name:
        raise ValidationError("Filename must not contain path separators.")

    suffix = Path(raw_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise ValidationError(
            f"Unsupported file type {suffix!r}. Allowed: {allowed}."
        )

    stem = slugify(Path(raw_name).stem)
    if not stem:
        raise ValidationError("Filename has no usable characters.")

    safe_name = f"{stem}{suffix}"

    size = getattr(uploaded_file, "size", None)
    limit = int(settings.MAX_CHECKPOINT_UPLOAD_BYTES)
    if size is not None and size > limit:
        raise ValidationError(
            f"File is too large ({_human_size(size)}); "
            f"limit is {_human_size(limit)}."
        )

    target = _root() / safe_name
    if target.exists():
        raise ValidationError(
            f"A checkpoint named {safe_name!r} already exists. "
            "Delete it first or rename the upload."
        )

    return safe_name


def save_uploaded_checkpoint(uploaded_file) -> Path:
    """Validate, write atomically, return the final path."""
    safe_name = validate_checkpoint_upload(uploaded_file)
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    final_path = root / safe_name

    tmp = tempfile.NamedTemporaryFile(
        dir=str(root),
        prefix=f"{safe_name}.",
        suffix=_PARTIAL_SUFFIX,
        delete=False,
    )
    tmp_path = Path(tmp.name)
    try:
        try:
            for chunk in uploaded_file.chunks():
                tmp.write(chunk)
        finally:
            tmp.close()
        if final_path.exists():
            raise ValidationError(
                f"A checkpoint named {safe_name!r} already exists."
            )
        os.replace(tmp_path, final_path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            logger.warning("Could not clean up partial upload %s", tmp_path)
        raise
    return final_path


def delete_checkpoint(name: str) -> bool:
    """Delete a checkpoint by name. Returns True on success.

    Refuses path separators, parent refs, symlinks, and any path
    that escapes `CHECKPOINT_DIR`.
    """
    if not name or name in (".", ".."):
        return False
    if "/" in name or "\\" in name:
        return False
    if name.startswith("."):
        return False

    root = _root()
    target = root / name
    if not _is_within(target, root):
        return False
    try:
        if target.is_symlink():
            return False
        if not target.exists():
            return False
        target.unlink()
    except OSError:
        return False
    return True
