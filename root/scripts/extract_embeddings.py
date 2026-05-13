#!/usr/bin/env python
"""Extract and cache patch embeddings from a pretrained Swin encoder.

Run once on GPU. Downstream clustering and statistical tests can then
run on CPU from the cached ``.npy`` file.

Usage:
    python scripts/extract_embeddings.py \
        --checkpoint outputs/my_run/best_model.pt \
        --data-root  ../../../data/patches_128 \
        --output     outputs/my_run/embeddings.npy

Output is a pickled dict with keys:
    Z             (N, D) float32 embedding matrix
    filenames     (N,)   patch filenames
    source_images (N,)   source image names
    image_indices (N,)   int64 image indices
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

# ---- path setup (same convention as notebooks) ----
_SCRIPT = Path(__file__).resolve()
_ROOT = _SCRIPT.parents[1]          # root/
_REPO = _ROOT.parent                # thesis/
for p in (_ROOT, _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from training.config import ModelCfg                # noqa: E402
from training.augment import ValSingleViewTransform # noqa: E402
from training.data import compute_channel_stats     # noqa: E402
from models.swin import build_swin_encoder          # noqa: E402
from utils_data.patch_dataset import PatchDataset   # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract and cache Swin encoder patch embeddings.",
    )
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to encoder checkpoint (.pt) with 'encoder_state_dict'.",
    )
    p.add_argument(
        "--data-root", type=str, default="../../../data/patches_128",
        help="Root directory of patch dataset (must contain index.csv).",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Output .npy path. Default: <checkpoint_dir>/embeddings.npy.",
    )
    p.add_argument(
        "--exclude-patterns", type=str, nargs="*", default=None,
        help="Source-image patterns to exclude (e.g. KONTROLA).",
    )
    p.add_argument(
        "--batch-size", type=int, default=64,
    )
    p.add_argument(
        "--num-workers", type=int, default=2,
    )
    p.add_argument(
        "--force", action="store_true",
        help="Recompute even if output already exists.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else ckpt_path.parent / "embeddings.npy"
    if out_path.exists() and not args.force:
        print(f"Cache already exists: {out_path}")
        print("Use --force to recompute.")
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- load checkpoint ----
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_cfg = ModelCfg()
    encoder = build_swin_encoder(model_cfg).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    n_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(f"Encoder loaded: {n_params:.2f}M params")

    # ---- channel statistics ----
    ch_mean = ckpt.get("channel_mean")
    ch_std = ckpt.get("channel_std")
    if ch_mean is not None:
        ch_mean = torch.tensor(ch_mean)
    if ch_std is not None:
        ch_std = torch.tensor(ch_std)

    # ---- dataset ----
    raw = PatchDataset(
        root=args.data_root,
        exclude_patterns=args.exclude_patterns,
    )
    print(f"Dataset: {len(raw)} patches from {args.data_root}")

    if ch_mean is None or ch_std is None:
        print("No channel stats in checkpoint, computing from data...")
        ch_mean, ch_std = compute_channel_stats(
            raw, in_channels=model_cfg.in_channels, max_samples=2048,
        )
    print(f"ch_mean={ch_mean.tolist()}")
    print(f"ch_std ={ch_std.tolist()}")

    transform = ValSingleViewTransform(ch_mean, ch_std)

    # ---- extract embeddings ----
    loader = torch.utils.data.DataLoader(
        raw, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type != "cpu"),
    )

    feats = []
    t0 = time.time()
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="embed", unit="batch"):
            x = batch if torch.is_tensor(batch) else batch[0]
            x = transform(x).to(device, non_blocking=True).contiguous()
            z = encoder(x)[-1].mean(dim=(-2, -1))   # global avg pool
            feats.append(z.float().cpu().numpy())

    Z = np.concatenate(feats, axis=0).astype(np.float32)
    elapsed = time.time() - t0
    print(f"Done: {Z.shape[0]} patches × {Z.shape[1]}D in {elapsed:.1f}s")

    # ---- gather metadata ----
    filenames = np.array([r["filename"] for r in raw.records])
    source_images = np.array([r["source_image"] for r in raw.records])
    image_indices = np.array(
        [int(r.get("image_index", 0)) for r in raw.records], dtype=np.int64,
    )

    # ---- save ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, {
        "Z": Z,
        "filenames": filenames,
        "source_images": source_images,
        "image_indices": image_indices,
    }, allow_pickle=True)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved: {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
