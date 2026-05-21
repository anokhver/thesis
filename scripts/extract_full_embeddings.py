#!/usr/bin/env python
"""Extract full patch embeddings into a self-contained bundle.

Mirrors notebooks/embeddings/extract_full_embeddings.ipynb but as a CLI.
Runs a frozen encoder over every patch in ``data/patches_128_from_zip/``
(all date subfolders) and writes a self-contained directory under
``data/embeddings/<run_name>/`` that downstream clustering / treatment-
group analyses can use without ever re-opening the original ``.npy``
patches.

Typical usage::

    python scripts/extract_full_embeddings.py \\
        --checkpoint data/training_outputs/pretrain_moby/<run>/best_model.pt

That's it — every other argument has a sensible default.

Output layout (``data/embeddings/<checkpoint-parent-folder-name>/``):

  embeddings.npy       float32 (N, D), plain np.save (mmap-friendly)
  metadata.csv         one row per patch, row i ⇔ embeddings.npy[i]
  manifest.json        checkpoint path/SHA, model cfg, channel stats, etc.
  group_patterns.json  treatment-group classification rules used
  README.md            inline doc + minimal load snippet
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
for _p in (_SRC, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from synaptic_ssl.training.config import ModelCfg                # noqa: E402
from synaptic_ssl.training.augment import ValSingleViewTransform # noqa: E402
from synaptic_ssl.training.data import TransformedSubset         # noqa: E402
from synaptic_ssl.models import build_swin_encoder, count_params # noqa: E402
from synaptic_ssl.utils_data.patch_dataset import PatchDataset   # noqa: E402
from synaptic_ssl.clustering.groups import (                     # noqa: E402
    load_group_rules,
    classify_image_by_patterns,
)


# ---------------------------------------------------------------------------
# Treatment-group classification
# ---------------------------------------------------------------------------
# The canonical pattern list lives in ``configs/clustering/group_patterns.json``
# (loaded via :func:`synaptic_ssl.clustering.groups.load_group_patterns`).
# This used to be a hard-coded 8-entry list in this file, which silently
# (a) routed PSIHARMIN files to "Psilocin", NORBAEOHARMIN files to
#     "Baeocystin + harmine", NORBAEO files to "Baeocystin", and
# (b) dropped LITHIUM / AERUG / AERUGHARMIN / standalone HARMIN files
#     into "UNKNOWN".
# Both bugs are gone now — the JSON config is the single source of truth.
DEFAULT_GROUP_PATTERNS_PATH = (
    _REPO / "configs" / "clustering" / "group_patterns.json"
)

# Default exclude list for the clustering extraction. This is the
# pretrain-style exclude list MINUS the KONTROLA entries, so that control
# patches survive the filter and end up in the embedding bundle (the clustering
# stage needs a control baseline). Pass --exclude-patterns-file to override or
# --no-exclude-patterns to disable filtering entirely.
DEFAULT_EXCLUDE_PATTERNS_PATH = (
    _REPO / "data" / "data_analysis" / "exclude_clustering.json"
)


def _resolve_group_patterns_path(arg: Path | None) -> Path:
    """Pick the group-patterns JSON: CLI arg → repo default."""
    if arg is not None:
        if not arg.is_file():
            raise FileNotFoundError(f"--group-patterns-file not found: {arg}")
        return arg
    if not DEFAULT_GROUP_PATTERNS_PATH.is_file():
        raise FileNotFoundError(
            f"Default group patterns file not found: "
            f"{DEFAULT_GROUP_PATTERNS_PATH}. Pass --group-patterns-file."
        )
    return DEFAULT_GROUP_PATTERNS_PATH


def _resolve_exclude_patterns_path(arg: Path | None) -> Path | None:
    """Pick the exclude-patterns JSON: CLI arg → repo default → None.

    Returns None only if the default file is missing AND no CLI arg was given.
    Raises FileNotFoundError if an explicit CLI path is broken.
    """
    if arg is not None:
        if not arg.is_file():
            raise FileNotFoundError(
                f"--exclude-patterns-file not found: {arg}"
            )
        return arg
    if DEFAULT_EXCLUDE_PATTERNS_PATH.is_file():
        return DEFAULT_EXCLUDE_PATTERNS_PATH
    return None


def classify_image(name: str, rules: dict) -> str | None:
    """Thin wrapper that forwards ``rules`` (from :func:`load_group_rules`)
    to :func:`classify_image_by_patterns`. Kept as a function so the
    per-patch loop below stays readable.
    """
    return classify_image_by_patterns(name, **rules)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_source_image(record: dict) -> str:
    val = record.get("source_image") or record.get("source_npy") or record.get("source_path")
    if not val:
        return "UNKNOWN"
    return str(val).replace("\\", "/").rsplit("/", 1)[-1] or str(val)


def _file_sha256(p: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_exclude_patterns(path: Path) -> list[str]:
    """Load exclude patterns from a JSON file.

    Mirrors the loader in ``synaptic_ssl.ssl_training.config_loading``: accepts
    either a plain JSON list of strings, or an object with key
    ``exclude_patterns``. Returns the list of patterns (case is preserved;
    matching is done case-insensitively inside ``PatchDataset``).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        patterns = raw.get("exclude_patterns", [])
    elif isinstance(raw, list):
        patterns = raw
    else:
        raise ValueError(
            f"{path} must be a JSON list or an object with key 'exclude_patterns'"
        )
    if not all(isinstance(s, str) for s in patterns):
        raise ValueError(f"{path}: every exclude pattern must be a string")
    return list(patterns)


# Filenames to try in priority order when a model directory is given.
_CKPT_PRIORITY = ("best_model.pt", "last.pt")


def resolve_checkpoint(path: Path) -> tuple[Path, str]:
    """Resolve a user-supplied path to a concrete checkpoint file.

    If ``path`` is a file, return it. If it is a directory, prefer
    ``best_model.pt``, then ``last.pt``, then the highest-numbered
    ``epoch_XXXX.pt``. Raises ``FileNotFoundError`` if none exist.

    Returns ``(checkpoint_path, selection_reason)``.
    """
    if path.is_file():
        return path, "explicit file"
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint path not found: {path}")

    for name in _CKPT_PRIORITY:
        cand = path / name
        if cand.is_file():
            return cand, f"auto-selected {name} from {path.name}/"

    epoch_ckpts = sorted(path.glob("epoch_*.pt"))
    if epoch_ckpts:
        latest = epoch_ckpts[-1]
        return latest, f"auto-selected latest epoch checkpoint {latest.name} from {path.name}/"

    raise FileNotFoundError(
        f"No checkpoint found in {path}. Looked for {list(_CKPT_PRIORITY)} "
        f"and epoch_*.pt."
    )


def load_encoder(ckpt_path: Path, device: torch.device):
    """Build Swin encoder and load weights from any of the supported ckpt formats."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_cfg = ModelCfg()
    encoder = build_swin_encoder(model_cfg).to(device)

    if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        load_source = "encoder_state_dict"
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        full_sd = ckpt["model_state_dict"]
        enc_sd = {k[len("swinViT."):]: v for k, v in full_sd.items() if k.startswith("swinViT.")}
        if not enc_sd:
            enc_sd = full_sd
        tgt = encoder.state_dict()
        matched = {k: v for k, v in enc_sd.items() if k in tgt and v.shape == tgt[k].shape}
        encoder.load_state_dict(matched, strict=False)
        load_source = f"model_state_dict ({len(matched)}/{len(tgt)} matched)"
    else:
        raw = ckpt if isinstance(ckpt, dict) else {}
        tgt = encoder.state_dict()
        matched = {k: v for k, v in raw.items() if k in tgt and v.shape == tgt[k].shape}
        encoder.load_state_dict(matched, strict=False)
        load_source = f"raw state_dict ({len(matched)}/{len(tgt)} matched)"

    encoder.eval()
    return encoder, ckpt, model_cfg, load_source


def resolve_channel_stats(ckpt: dict, ckpt_path: Path) -> tuple[torch.Tensor, torch.Tensor, str]:
    if isinstance(ckpt, dict) and "channel_mean" in ckpt and "channel_std" in ckpt:
        return (
            torch.tensor(ckpt["channel_mean"], dtype=torch.float32),
            torch.tensor(ckpt["channel_std"],  dtype=torch.float32),
            "checkpoint",
        )
    side = ckpt_path.parent / "channel_stats.json"
    if side.exists():
        with side.open() as f:
            cs = json.load(f)
        m = cs.get("channel_mean", cs.get("mean"))
        s = cs.get("channel_std",  cs.get("std"))
        return (
            torch.tensor(m, dtype=torch.float32),
            torch.tensor(s, dtype=torch.float32),
            f"sidecar {side.name}",
        )
    raise RuntimeError(
        "No channel_mean/channel_std in checkpoint and no channel_stats.json "
        "next to checkpoint. Cannot normalise inputs."
    )


@torch.no_grad()
def encode_dataset(encoder, dataset, *, device, batch_size: int, num_workers: int) -> np.ndarray:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    chunks = []
    for batch in tqdm(loader, leave=False, desc="  batches"):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True).contiguous()
        z = encoder(x)[-1].mean(dim=(-2, -1))
        chunks.append(z.float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract full patch embeddings into a self-contained bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", "-c", type=Path, required=True,
        help=(
            "Path to an encoder checkpoint (.pt) OR a model run directory. "
            "If a directory is given, the script auto-picks best_model.pt, "
            "falling back to last.pt, then the highest-numbered epoch_*.pt."
        ),
    )
    p.add_argument(
        "--data-root", type=Path, default=_REPO / "data" / "patches_128_from_zip",
        help="Root containing <date>/index.csv subfolders.",
    )
    p.add_argument(
        "--output-root", type=Path, default=_REPO / "data" / "embeddings",
        help="Parent directory; bundle is written to <output-root>/<run-name>/.",
    )
    p.add_argument(
        "--run-name", type=str, default=None,
        help="Override bundle folder name (default: model run directory name).",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--exclude-dates", type=str, nargs="*", default=[],
        help="Skip these date subfolders by name.",
    )
    p.add_argument(
        "--exclude-patterns-file", type=Path, default=None,
        help="JSON file with case-insensitive substring patterns to drop from "
             "the patch index (matched against source_image / source_path / "
             "source_npy). Accepts a JSON list or an object with key "
             "'exclude_patterns'. Default: "
             "data/data_analysis/exclude_clustering.json (pretrain exclude "
             "list minus the KONTROLA entries, so controls survive for the "
             "clustering baseline). Pass --no-exclude-patterns to disable.",
    )
    p.add_argument(
        "--no-exclude-patterns", action="store_true",
        help="Disable the default exclude file; keep every patch in the index.",
    )
    p.add_argument(
        "--group-patterns-file", type=Path, default=None,
        help="JSON file with the treatment-group substring patterns. "
             "Default: configs/clustering/group_patterns.json. The same file "
             "is consumed by scripts/run_clustering.py so both stages agree.",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Force device (cuda / cpu). Default: cuda if available.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing bundle files; default: refuse if embeddings.npy exists.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    input_path: Path = args.checkpoint.resolve()
    data_root:  Path = args.data_root.resolve()
    if not input_path.exists():
        print(f"ERROR: checkpoint path not found: {input_path}", file=sys.stderr)
        return 2
    if not data_root.exists():
        print(f"ERROR: data root not found: {data_root}", file=sys.stderr)
        return 2

    try:
        ckpt_path, ckpt_selection = resolve_checkpoint(input_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Run name = model directory name (the parent folder of the .pt file),
    # so passing either the directory or a .pt inside it yields the same bundle.
    run_name = args.run_name or ckpt_path.parent.name
    out_dir: Path = (args.output_root / run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / "embeddings.npy"
    if emb_path.exists() and not args.overwrite:
        print(f"ERROR: {emb_path} already exists. Pass --overwrite to replace.", file=sys.stderr)
        return 2

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"input path : {input_path}")
    print(f"checkpoint : {ckpt_path}  ({ckpt_selection})")
    print(f"data root  : {data_root}")
    print(f"output dir : {out_dir}")
    print(f"device     : {device}")
    print()

    # ── Encoder + channel stats ──────────────────────────────────────────
    print("[1/4] Loading checkpoint & building encoder…")
    encoder, ckpt, model_cfg, load_source = load_encoder(ckpt_path, device)
    print(f"      loaded via {load_source}; params = {count_params(encoder)/1e6:.2f} M")

    ch_mean_t, ch_std_t, stats_source = resolve_channel_stats(ckpt, ckpt_path)
    print(f"      channel_mean = {ch_mean_t.tolist()} (from {stats_source})")
    print(f"      channel_std  = {ch_std_t.tolist()}")
    transform = ValSingleViewTransform(ch_mean=ch_mean_t, ch_std=ch_std_t)

    # ── Discover date subfolders ─────────────────────────────────────────
    exclude = set(args.exclude_dates)
    date_dirs = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and (d / "index.csv").exists() and d.name not in exclude
    )
    if not date_dirs:
        print(f"ERROR: no date subfolders with index.csv under {data_root}", file=sys.stderr)
        return 2
    print(f"\n[2/4] Found {len(date_dirs)} date subfolders:")
    for d in date_dirs:
        print(f"      {d.name}")

    # ── Load patch-pattern exclusion list (matches pretrain config) ─────
    # Default: data/data_analysis/exclude_clustering.json (keeps KONTROLA so
    # controls survive for the clustering baseline). Override with
    # --exclude-patterns-file, or disable entirely with --no-exclude-patterns.
    exclude_patterns: list[str] = []
    exclude_patterns_source: str | None = None
    if args.no_exclude_patterns:
        if args.exclude_patterns_file is not None:
            print(
                "ERROR: --no-exclude-patterns and --exclude-patterns-file are "
                "mutually exclusive.",
                file=sys.stderr,
            )
            return 2
        print("      exclude list: DISABLED (--no-exclude-patterns)")
    else:
        try:
            epf = _resolve_exclude_patterns_path(args.exclude_patterns_file)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        if epf is None:
            print(
                f"      exclude list: none (default "
                f"{DEFAULT_EXCLUDE_PATTERNS_PATH} missing and no CLI override)"
            )
        else:
            epf = epf.resolve()
            exclude_patterns = _load_exclude_patterns(epf)
            exclude_patterns_source = str(epf)
            tag = "default" if args.exclude_patterns_file is None else "override"
            print(
                f"      excluding {len(exclude_patterns)} substring patterns "
                f"loaded from {epf}  ({tag})"
            )

    # ── Resolve group-patterns file (canonical treatment-group rules) ───
    group_patterns_path = _resolve_group_patterns_path(args.group_patterns_file)
    group_rules = load_group_rules(group_patterns_path)
    print(
        f"      group patterns: {group_patterns_path}  "
        f"({len(group_rules['patterns'])} bases, "
        f"{len(group_rules['modifiers'])} modifiers, "
        f"{len(group_rules['concentrations'])} concentrations, "
        f"{len(group_rules['time_points'])} time-points, "
        f"{len(group_rules['washouts'])} washouts)"
    )

    # ── Encode every date ────────────────────────────────────────────────
    print("\n[3/4] Extracting embeddings…")
    all_Z: list[np.ndarray] = []
    all_rows: list[dict] = []
    embed_dim: int | None = None
    t0 = time.time()

    for d in date_dirs:
        raw_ds = PatchDataset(root=d, exclude_patterns=exclude_patterns)
        tx_ds  = TransformedSubset(raw_ds, transform=transform)
        print(f"  [{d.name}] {len(raw_ds)} patches")

        Z = encode_dataset(
            encoder, tx_ds,
            device=device, batch_size=args.batch_size, num_workers=args.num_workers,
        )
        if embed_dim is None:
            embed_dim = int(Z.shape[1])
        elif Z.shape[1] != embed_dim:
            print(f"ERROR: embed-dim mismatch in {d.name}: {Z.shape[1]} vs {embed_dim}", file=sys.stderr)
            return 3

        records = raw_ds.records
        if len(records) != Z.shape[0]:
            print(
                f"ERROR: record/feature length mismatch in {d.name}: "
                f"{len(records)} vs {Z.shape[0]}",
                file=sys.stderr,
            )
            return 3

        for r in records:
            fname = r["filename"]
            src   = _resolve_source_image(r)
            group = (
                classify_image(src, group_rules)
                or classify_image(fname, group_rules)
            )
            all_rows.append({
                "date":            d.name,
                "filename":        fname,
                "source_image":    src,
                "treatment_group": group if group is not None else "UNKNOWN",
                "grid_row":        r.get("grid_row", ""),
                "grid_col":        r.get("grid_col", ""),
                "mean_intensity":  r.get("mean_intensity", ""),
                "channels":        r.get("channels", ""),
                "patch_size":      r.get("patch_size", ""),
                "source_path":     r.get("source_path", ""),
            })
        all_Z.append(Z)

    Z_full = np.concatenate(all_Z, axis=0).astype(np.float32)
    N, D = Z_full.shape
    print(f"\n      total: N={N} patches, D={D}  ({time.time()-t0:.1f}s)")

    # ── Save bundle ──────────────────────────────────────────────────────
    print("\n[4/4] Writing bundle…")
    np.save(emb_path, Z_full)
    print(f"      wrote {emb_path}  ({emb_path.stat().st_size/1e6:.1f} MB)")

    meta_path = out_dir / "metadata.csv"
    field_order = [
        "embedding_idx",
        "date",
        "filename",
        "source_image",
        "treatment_group",
        "grid_row", "grid_col",
        "mean_intensity", "channels", "patch_size",
        "source_path",
    ]
    with meta_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        for i, row in enumerate(all_rows):
            writer.writerow({"embedding_idx": i,
                             **{k: row.get(k, "") for k in field_order if k != "embedding_idx"}})
    print(f"      wrote {meta_path}  ({meta_path.stat().st_size/1e6:.2f} MB, {len(all_rows)} rows)")

    model_cfg_dict = asdict(model_cfg) if is_dataclass(model_cfg) else dict(vars(model_cfg))
    try:
        ckpt_rel = str(ckpt_path.relative_to(_REPO))
    except ValueError:
        ckpt_rel = str(ckpt_path)
    try:
        data_rel = str(data_root.relative_to(_REPO))
    except ValueError:
        data_rel = str(data_root)

    manifest = {
        "created_at":       datetime.now().isoformat(timespec="seconds"),
        "run_name":         run_name,
        "checkpoint": {
            "path":         ckpt_rel,
            "abs_path":     str(ckpt_path),
            "sha256":       _file_sha256(ckpt_path),
            "load_source":  load_source,
            "selection":    ckpt_selection,
        },
        "data_root":        data_rel,
        "date_folders":     [d.name for d in date_dirs],
        "excluded_dates":   sorted(exclude),
        "exclude_patterns_file":  exclude_patterns_source,
        "exclude_patterns":       exclude_patterns,
        "total_patches":    int(N),
        "embedding_dim":    int(D),
        "embedding_dtype":  str(Z_full.dtype),
        "channel_mean":     ch_mean_t.tolist(),
        "channel_std":      ch_std_t.tolist(),
        "channel_stats_source": stats_source,
        "model_cfg":        model_cfg_dict,
        "pooling":          "global average pool over last encoder stage (H, W)",
        "files": {
            "embeddings":      "embeddings.npy",
            "metadata":        "metadata.csv",
            "group_patterns":  "group_patterns.json",
        },
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"      wrote {out_dir / 'manifest.json'}")

    # Snapshot the *full* group-patterns JSON (all sections: patterns,
    # modifiers, concentrations, time_points, washouts) so this bundle
    # is reproducible without depending on the repo checkout.
    with open(group_patterns_path) as _src, \
            (out_dir / "group_patterns.json").open("w") as _dst:
        _dst.write(_src.read())
    print(f"      wrote {out_dir / 'group_patterns.json'}")

    readme = f"""# Embeddings bundle: {run_name}

Self-contained patch-embedding artefact. **Original `.npy` patches are NOT
required** to use this folder.

## Files

| file                  | description                                              |
|-----------------------|----------------------------------------------------------|
| `embeddings.npy`      | float32 array, shape (N, D) = ({N}, {D})                    |
| `metadata.csv`        | one row per patch; row `i` ⇔ `embeddings.npy[i]`         |
| `manifest.json`       | checkpoint path/SHA, model cfg, channel stats            |
| `group_patterns.json` | rules used to derive `metadata.csv:treatment_group`      |

## Minimal load example

```python
import numpy as np, pandas as pd
Z   = np.load('embeddings.npy', mmap_mode='r')          # (N, D)
df  = pd.read_csv('metadata.csv')                       # N rows
assert len(df) == Z.shape[0]
print(df['treatment_group'].value_counts())
```

Created {datetime.now().isoformat(timespec="seconds")}.
"""
    (out_dir / "README.md").write_text(readme)
    print(f"      wrote {out_dir / 'README.md'}")

    # ── Summary ──────────────────────────────────────────────────────────
    date_counts  = Counter(r["date"] for r in all_rows)
    group_counts = Counter(r["treatment_group"] for r in all_rows)
    print("\nPatches per date:")
    for k, v in sorted(date_counts.items()):
        print(f"  {k:<12} {v:>7d}")
    print("\nPatches per treatment group:")
    for k, v in sorted(group_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<28} {v:>7d}")

    print(f"\nDone → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
