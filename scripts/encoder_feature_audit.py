#!/usr/bin/env python
"""Encoder/projector feature-quality audit across SSL pretraining checkpoints.

For every saved SimMIM+VICReg pretraining run, load the chosen checkpoint
(default ``last.pt``), run a fixed sample of patches through the encoder and
the VICReg projector, and report three representation-quality diagnostics:

* ``mean_per_dim_std``  — average per-dimension standard deviation of the
  feature matrix (collapse warning when it approaches zero).
* ``mean_abs_corr``     — mean absolute Pearson correlation across the strict
  upper triangle of the feature correlation matrix (decorrelation indicator;
  VICReg's covariance term directly targets this).
* ``effective_rank``    — RankMe-style soft rank (Garrido et al., ICML 2023):
  with singular values ``σ`` of mean-centred features and
  ``p = σ² / Σσ²``, ``rank = exp(-Σ p log p)``.

All twelve runs (3 inits × 4 variants) are evaluated on the same fixed set of
patches, normalised per-checkpoint with the checkpoint's own channel mean/std.

Usage
-----
.. code-block:: bash

    python scripts/encoder_feature_audit.py \
        [--ckpt_name last.pt] [--n_samples 8000] \
        [--device cuda|cpu] [--repo_root <repo>] \
        [--data_root <repo>/data/patches_128] \
        [--runs_root <repo>/../models/training_outputs] \
        [--out_dir <runs_root>/encoder_audit]

Outputs (under ``--out_dir``):

* ``encoder_audit.csv`` — one row per checkpoint, columns:
  ``run_dir, init_source, variant_tag, w_recon, w_vicreg, w_fourier,
  lambda_sim, lambda_std, lambda_cov, ckpt_epoch, val_recon_fg_at_best,
  enc_mean_per_dim_std, enc_mean_abs_corr, enc_effective_rank,
  proj_mean_per_dim_std, proj_mean_abs_corr, proj_effective_rank,
  n_samples, ckpt_path``.
* ``encoder_audit.md`` — one section per ``variant_tag`` (4 sections), each
  showing the 3 inits side-by-side.
* ``run.log`` — per-checkpoint diagnostics, parameter counts, runtime,
  warnings.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


# --------------------------------------------------------------------------- #
# Repo / sys.path discovery
# --------------------------------------------------------------------------- #
def _find_repo_root(start: Path) -> Path:
    """Walk parents of ``start`` until one contains ``.git``.

    Falls back to ``Path(__file__).resolve().parents[1]`` if no marker is
    found (matches the existing convention in ``scripts/extract_embeddings.py``).
    """
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    return Path(__file__).resolve().parents[1]


_REPO = _find_repo_root(Path(__file__).resolve())
_SRC = _REPO / "src"
for _p in (_SRC, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from synaptic_ssl.models.swin import (                       # noqa: E402
    build_simmim_vicreg_heads,
    build_swin_encoder,
)
from synaptic_ssl.training.augment import ValSingleViewTransform  # noqa: E402
from synaptic_ssl.training.config import ModelCfg, SSLCfg    # noqa: E402
from synaptic_ssl.training.data import TransformedSubset     # noqa: E402
from synaptic_ssl.training.seeding import seed_everything    # noqa: E402
from synaptic_ssl.utils_data.patch_dataset import PatchDataset  # noqa: E402


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Audit encoder + projector feature quality across SSL "
            "pretraining checkpoints."
        )
    )
    p.add_argument(
        "--ckpt_name", type=str, default="last.pt",
        help="Checkpoint filename inside each run directory.",
    )
    p.add_argument(
        "--n_samples", type=int, default=8000,
        help="Number of patches to sample (fixed across all checkpoints).",
    )
    p.add_argument(
        "--device", type=str, default=None,
        choices=["cuda", "cpu"],
        help="Compute device. Default: cuda if available, else cpu.",
    )
    p.add_argument(
        "--repo_root", type=str, default=None,
        help="Override repository root.",
    )
    p.add_argument(
        "--data_root", type=str, default=None,
        help="Patch dataset root (must contain index.csv). "
             "Default: <repo_root>/data/patches_128.",
    )
    p.add_argument(
        "--runs_root", type=str, default=None,
        help="Directory containing pretrain_* subdirs. "
             "Default: <repo_root>/../models/training_outputs.",
    )
    p.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory. Default: <runs_root>/encoder_audit.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Diagnostic primitives (float64 throughout)
# --------------------------------------------------------------------------- #
def _filter_dataclass_kwargs(cls, src: dict) -> dict:
    """Return ``src`` restricted to the field names of dataclass ``cls``."""
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in src.items() if k in names}


def _coerce_model_cfg(src: dict) -> ModelCfg:
    kwargs = _filter_dataclass_kwargs(ModelCfg, src)
    if "depths" in kwargs and isinstance(kwargs["depths"], list):
        kwargs["depths"] = tuple(kwargs["depths"])
    if "num_heads" in kwargs and isinstance(kwargs["num_heads"], list):
        kwargs["num_heads"] = tuple(kwargs["num_heads"])
    return ModelCfg(**kwargs)


def _coerce_ssl_cfg(src: dict) -> SSLCfg:
    return SSLCfg(**_filter_dataclass_kwargs(SSLCfg, src))


def mean_per_dim_std(Z: np.ndarray) -> float:
    """Mean over feature dimensions of per-dimension std (ddof=1)."""
    if Z.shape[0] < 2:
        return float("nan")
    return float(Z.std(axis=0, ddof=1).mean())


def mean_abs_pairwise_corr(Z: np.ndarray, eps: float = 1e-6) -> float:
    """Mean ``|ρ_ij|`` over the strict upper triangle of ``corrcoef(Z)``.

    Dimensions with std < ``eps`` are masked: their rows/cols are set to
    NaN before averaging, and ``np.nanmean`` ignores them. Returns NaN when
    all pairs are masked.
    """
    if Z.shape[0] < 2 or Z.shape[1] < 2:
        return float("nan")
    stds = Z.std(axis=0, ddof=1)
    bad = stds < eps
    # np.corrcoef of a constant column yields NaN entries already; mask
    # explicitly so we control behaviour.
    C = np.corrcoef(Z, rowvar=False)
    if bad.any():
        C[bad, :] = np.nan
        C[:, bad] = np.nan
    D = C.shape[0]
    iu = np.triu_indices(D, k=1)
    vals = np.abs(C[iu])
    if np.all(np.isnan(vals)):
        return float("nan")
    return float(np.nanmean(vals))


def effective_rank(Z: np.ndarray) -> float:
    """RankMe-style effective rank: ``exp(-Σ p log p)`` with ``p = σ²/Σσ²``.

    ``σ`` are the singular values of mean-centred ``Z``. Zero-mass entries
    are skipped to avoid ``0 * log 0`` NaNs.
    """
    if Z.shape[0] < 2 or Z.shape[1] < 1:
        return float("nan")
    Zc = Z - Z.mean(axis=0, keepdims=True)
    # SVD is more numerically stable than eig of Z^T Z for tall matrices.
    s = np.linalg.svd(Zc, compute_uv=False)
    s2 = s ** 2
    total = s2.sum()
    if total <= 0.0 or not np.isfinite(total):
        return float("nan")
    p = s2 / total
    p = p[p > 0.0]
    if p.size == 0:
        return float("nan")
    entropy = -float((p * np.log(p)).sum())
    return float(np.exp(entropy))


# --------------------------------------------------------------------------- #
# Channel-stats loader
# --------------------------------------------------------------------------- #
def _load_channel_stats(
    ckpt: dict, run_dir: Path, log: logging.Logger,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve channel mean/std from the checkpoint dict or sibling JSON."""
    ch_mean = ckpt.get("channel_mean")
    ch_std = ckpt.get("channel_std")
    if ch_mean is not None and ch_std is not None:
        return (
            torch.as_tensor(ch_mean, dtype=torch.float32),
            torch.as_tensor(ch_std, dtype=torch.float32),
        )
    stats_path = run_dir / "channel_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"No channel_mean/channel_std in checkpoint and "
            f"{stats_path} missing."
        )
    log.info("Falling back to %s for channel stats", stats_path.name)
    stats = json.loads(stats_path.read_text())
    return (
        torch.as_tensor(stats["mean"], dtype=torch.float32),
        torch.as_tensor(stats["std"], dtype=torch.float32),
    )


# --------------------------------------------------------------------------- #
# metrics.csv -> best_val_metric on the final logged row
# --------------------------------------------------------------------------- #
def _read_best_val_metric(metrics_csv: Path) -> float:
    if not metrics_csv.exists():
        return float("nan")
    last_row: dict | None = None
    with metrics_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            last_row = row
    if last_row is None or "best_val_metric" not in last_row:
        return float("nan")
    try:
        return float(last_row["best_val_metric"])
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- #
# Per-checkpoint evaluation
# --------------------------------------------------------------------------- #
def _eval_checkpoint(
    *,
    ckpt_path: Path,
    raw_subset: Subset,
    device: torch.device,
    batch_size: int,
    log: logging.Logger,
) -> dict:
    """Load a checkpoint, run the fixed subset through encoder+projector,
    and return diagnostics + parsed config provenance."""
    run_dir = ckpt_path.parent
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    cfg_raw = json.loads(config_path.read_text())
    base_cfg = cfg_raw.get("base", {})
    model_cfg = _coerce_model_cfg(cfg_raw.get("model", {}))
    ssl_cfg = _coerce_ssl_cfg(cfg_raw.get("ssl", {}))

    # variant_tag = everything after the first underscore of base.tag
    tag = str(base_cfg.get("tag", ""))
    variant_tag = tag.split("_", 1)[1] if "_" in tag else tag
    init_source = str(base_cfg.get("init_source", "unknown"))

    # ---- build models ----
    encoder = build_swin_encoder(model_cfg).to(device)
    heads = build_simmim_vicreg_heads(model_cfg, ssl_cfg)
    projector = heads["projector"].to(device)

    enc_params = sum(p.numel() for p in encoder.parameters())
    proj_params = sum(p.numel() for p in projector.parameters())
    log.info(
        "  encoder=%.2fM params  projector=%.2fM params",
        enc_params / 1e6, proj_params / 1e6,
    )

    # ---- load weights ----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc_load = encoder.load_state_dict(
        ckpt["encoder_state_dict"], strict=False,
    )
    if enc_load.missing_keys:
        log.warning("  encoder missing_keys=%d", len(enc_load.missing_keys))
    if enc_load.unexpected_keys:
        log.warning(
            "  encoder unexpected_keys=%d", len(enc_load.unexpected_keys),
        )

    heads_state = ckpt.get("heads_state_dict") or {}
    proj_state = heads_state.get("projector")
    projector_loaded = False
    if proj_state is not None:
        proj_load = projector.load_state_dict(proj_state, strict=False)
        if proj_load.missing_keys:
            log.warning(
                "  projector missing_keys=%d", len(proj_load.missing_keys),
            )
        if proj_load.unexpected_keys:
            log.warning(
                "  projector unexpected_keys=%d",
                len(proj_load.unexpected_keys),
            )
        projector_loaded = True
    else:
        log.warning(
            "  No projector state in checkpoint; proj_* will be NaN."
        )

    encoder.eval()
    projector.eval()

    # ---- per-checkpoint channel stats ----
    ch_mean, ch_std = _load_channel_stats(ckpt, run_dir, log)
    transform = ValSingleViewTransform(ch_mean, ch_std)
    dataset = TransformedSubset(raw_subset, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type != "cpu"),
    )

    stage_idx = int(ssl_cfg.head_stage_index)
    enc_chunks: list[np.ndarray] = []
    proj_chunks: list[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            x = batch if torch.is_tensor(batch) else batch[0]
            x = x.to(device, non_blocking=True).contiguous()
            z_stage = encoder(x)[stage_idx]
            z_enc = z_stage.float().mean(dim=(-2, -1))
            enc_chunks.append(z_enc.detach().cpu().numpy())
            if projector_loaded:
                z_proj = projector(z_enc).float()
                proj_chunks.append(z_proj.detach().cpu().numpy())
    fwd_time = time.time() - t0

    Z_enc = np.concatenate(enc_chunks, axis=0).astype(np.float64, copy=False)
    log.info(
        "  features encoder=%s  fwd_time=%.1fs",
        Z_enc.shape, fwd_time,
    )

    enc_std = mean_per_dim_std(Z_enc)
    enc_corr = mean_abs_pairwise_corr(Z_enc)
    enc_rank = effective_rank(Z_enc)

    if projector_loaded:
        Z_proj = np.concatenate(proj_chunks, axis=0).astype(
            np.float64, copy=False,
        )
        proj_std = mean_per_dim_std(Z_proj)
        proj_corr = mean_abs_pairwise_corr(Z_proj)
        proj_rank = effective_rank(Z_proj)
    else:
        proj_std = proj_corr = proj_rank = float("nan")

    val_best = _read_best_val_metric(run_dir / "metrics.csv")

    return {
        "run_dir": str(run_dir),
        "init_source": init_source,
        "variant_tag": variant_tag,
        "w_recon": float(ssl_cfg.w_recon),
        "w_vicreg": float(ssl_cfg.w_vicreg),
        "w_fourier": float(ssl_cfg.w_fourier),
        "lambda_sim": float(ssl_cfg.lambda_sim),
        "lambda_std": float(ssl_cfg.lambda_std),
        "lambda_cov": float(ssl_cfg.lambda_cov),
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "val_recon_fg_at_best": val_best,
        "enc_mean_per_dim_std": enc_std,
        "enc_mean_abs_corr": enc_corr,
        "enc_effective_rank": enc_rank,
        "proj_mean_per_dim_std": proj_std,
        "proj_mean_abs_corr": proj_corr,
        "proj_effective_rank": proj_rank,
        "n_samples": int(Z_enc.shape[0]),
        "ckpt_path": str(ckpt_path),
    }


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
_CSV_COLUMNS = [
    "run_dir", "init_source", "variant_tag",
    "w_recon", "w_vicreg", "w_fourier",
    "lambda_sim", "lambda_std", "lambda_cov",
    "ckpt_epoch", "val_recon_fg_at_best",
    "enc_mean_per_dim_std", "enc_mean_abs_corr", "enc_effective_rank",
    "proj_mean_per_dim_std", "proj_mean_abs_corr", "proj_effective_rank",
    "n_samples", "ckpt_path",
]


def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _CSV_COLUMNS})


def _fmt(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NaN"
    return f"{x:.{digits}f}"


def _write_markdown(
    rows: list[dict], path: Path, *,
    n_samples: int, ckpt_name: str, device: str, total_runtime_s: float,
) -> None:
    lines: list[str] = []
    lines.append("# Encoder feature audit\n")
    lines.append(
        f"- samples per checkpoint: **{n_samples}** "
        f"(same fixed patch indices for every run)\n"
        f"- checkpoint file: `{ckpt_name}`\n"
        f"- device: `{device}`\n"
        f"- total runtime: {total_runtime_s:.1f}s\n"
    )
    lines.append("\n**Diagnostic formulae**\n")
    lines.append(
        "- `σ̄` = mean over feature dims of per-dim std "
        "(`Z.std(0, ddof=1).mean()`).\n"
    )
    lines.append(
        "- `ρ̄` = mean `|ρ_ij|` over strict upper triangle of "
        "`corrcoef(Z)`; dims with std<1e-6 are masked.\n"
    )
    lines.append(
        "- `rank` = `exp(-Σ p log p)` with `p = σ²/Σσ²` and `σ` the "
        "singular values of mean-centred `Z` (RankMe; Garrido et al. 2023).\n"
    )

    # group by variant_tag → 4 sections, each with 3 init rows
    variants: dict[str, list[dict]] = {}
    for row in rows:
        variants.setdefault(row["variant_tag"], []).append(row)

    _init_order = {"scratch": 0, "moby": 1, "tinny22k": 2}

    for variant in sorted(variants.keys()):
        bucket = sorted(
            variants[variant],
            key=lambda r: _init_order.get(r["init_source"], 99),
        )
        lines.append(f"\n## variant: `{variant}`\n")
        lines.append(
            "| init | enc σ̄ | enc ρ̄ | enc rank | "
            "proj σ̄ | proj ρ̄ | proj rank |\n"
        )
        lines.append("|---|---|---|---|---|---|---|\n")
        for row in bucket:
            lines.append(
                "| {init} | {es} | {ec} | {er} | "
                "{ps} | {pc} | {pr} |\n".format(
                    init=row["init_source"],
                    es=_fmt(row["enc_mean_per_dim_std"]),
                    ec=_fmt(row["enc_mean_abs_corr"]),
                    er=_fmt(row["enc_effective_rank"], 2),
                    ps=_fmt(row["proj_mean_per_dim_std"]),
                    pc=_fmt(row["proj_mean_abs_corr"]),
                    pr=_fmt(row["proj_effective_rank"], 2),
                )
            )

    path.write_text("".join(lines))


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
def _run_sanity_checks(rows: list[dict], log: logging.Logger) -> None:
    # 1. all enc_mean_per_dim_std finite and > 0
    for row in rows:
        v = row["enc_mean_per_dim_std"]
        if not np.isfinite(v) or v <= 0.0:
            log.warning(
                "SANITY: enc_mean_per_dim_std=%s for %s (expected > 0)",
                v, row["run_dir"],
            )

    # 2. For each (init, *_vicreg_on / *_vicreg_weak): proj decorrelation
    #    should beat the matching *_vicreg_off baseline (lower mean |ρ|).
    by_init_variant = {
        (r["init_source"], r["variant_tag"]): r for r in rows
    }
    # Identify the "vicreg-off" baseline for each init. By design the
    # `no_fourier_vicreg_off` variant has ``w_vicreg == 0``.
    for (init, variant), row in by_init_variant.items():
        if row["w_vicreg"] <= 0.0:
            continue  # only check rows where VICReg is on
        # find the matching vicreg-off baseline at the same init
        off = next(
            (
                r for r in rows
                if r["init_source"] == init and r["w_vicreg"] <= 0.0
            ),
            None,
        )
        if off is None:
            continue
        if not (
            np.isfinite(row["proj_mean_abs_corr"])
            and np.isfinite(off["proj_mean_abs_corr"])
        ):
            continue
        if row["proj_mean_abs_corr"] >= off["proj_mean_abs_corr"]:
            log.warning(
                "SANITY: %s/%s proj_mean_abs_corr=%.4f is not below "
                "vicreg-off baseline (%s) %.4f",
                init, variant, row["proj_mean_abs_corr"],
                off["variant_tag"], off["proj_mean_abs_corr"],
            )

    # 3. For scratch init only: proj_mean_per_dim_std(vicreg_off) should
    #    be LESS than proj_mean_per_dim_std(vicreg_on) — the variance
    #    hinge actively pushes std away from zero.
    scratch_off = next(
        (
            r for r in rows
            if r["init_source"] == "scratch" and r["w_vicreg"] <= 0.0
        ),
        None,
    )
    scratch_on = next(
        (
            r for r in rows
            if r["init_source"] == "scratch" and r["w_vicreg"] > 0.0
        ),
        None,
    )
    if scratch_off is not None and scratch_on is not None:
        a = scratch_off["proj_mean_per_dim_std"]
        b = scratch_on["proj_mean_per_dim_std"]
        if np.isfinite(a) and np.isfinite(b) and a >= b:
            log.warning(
                "SANITY: scratch proj_mean_per_dim_std vicreg_off=%.4f "
                ">= vicreg_on=%.4f (expected vicreg_off < vicreg_on).",
                a, b,
            )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _REPO
    data_root = (
        Path(args.data_root).resolve()
        if args.data_root else repo_root / "data" / "patches_128"
    )
    runs_root = (
        Path(args.runs_root).resolve()
        if args.runs_root else (repo_root.parent / "models" / "training_outputs")
    )
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir else runs_root / "encoder_audit"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # device
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: --device cuda requested but no CUDA device available.",
                  file=sys.stderr)
            return 1
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # logger
    log_path = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    log = logging.getLogger("encoder_audit")

    # banner
    log.info("repo_root  = %s", repo_root)
    log.info("data_root  = %s", data_root)
    log.info("runs_root  = %s", runs_root)
    log.info("out_dir    = %s", out_dir)
    log.info(
        "device     = %s (%s)",
        device,
        torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    )
    log.info("torch      = %s", torch.__version__)
    log.info("numpy      = %s", np.__version__)

    # ---- input validation ----
    if not data_root.exists():
        log.error("data_root does not exist: %s", data_root)
        return 1
    if not runs_root.exists():
        log.error("runs_root does not exist: %s", runs_root)
        return 1
    pretrain_dirs = sorted(runs_root.glob("pretrain_*"))
    if not pretrain_dirs:
        log.error(
            "No pretrain_* subdirectories under runs_root=%s", runs_root,
        )
        return 1

    # ---- determinism ----
    seed_everything(0)
    rng = np.random.RandomState(0)

    # ---- discover checkpoints ----
    ckpts: list[Path] = []
    for pdir in pretrain_dirs:
        ckpts.extend(sorted(pdir.glob(f"simmim_vicreg_pretrain_*/{args.ckpt_name}")))
    if not ckpts:
        log.error(
            "No checkpoints matching %s under %s/pretrain_*/", args.ckpt_name, runs_root,
        )
        return 1
    log.info("Discovered %d checkpoint(s):", len(ckpts))
    for c in ckpts:
        log.info("  %s", c.relative_to(runs_root))

    # ---- build the shared raw dataset & fixed index set ----
    raw_dataset = PatchDataset(root=str(data_root))
    if len(raw_dataset) == 0:
        log.error("PatchDataset is empty at %s", data_root)
        return 1
    n_samples = min(args.n_samples, len(raw_dataset))
    if n_samples < args.n_samples:
        log.warning(
            "Requested %d samples but dataset has only %d; using %d.",
            args.n_samples, len(raw_dataset), n_samples,
        )
    fixed_indices = rng.choice(
        len(raw_dataset), size=n_samples, replace=False,
    )
    fixed_indices.sort()
    log.info(
        "Sampled %d patches from %d (seed=0).",
        n_samples, len(raw_dataset),
    )
    raw_subset = Subset(raw_dataset, fixed_indices.tolist())

    # ---- evaluate each checkpoint ----
    rows: list[dict] = []
    t_all = time.time()
    for i, ckpt_path in enumerate(ckpts, 1):
        log.info(
            "[%d/%d] %s", i, len(ckpts),
            ckpt_path.relative_to(runs_root),
        )
        try:
            row = _eval_checkpoint(
                ckpt_path=ckpt_path,
                raw_subset=raw_subset,
                device=device,
                batch_size=128,
                log=log,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("  FAILED: %s", exc)
            continue
        log.info(
            "  enc: σ̄=%.4f ρ̄=%.4f rank=%.2f | "
            "proj: σ̄=%.4f ρ̄=%.4f rank=%.2f",
            row["enc_mean_per_dim_std"], row["enc_mean_abs_corr"],
            row["enc_effective_rank"],
            row["proj_mean_per_dim_std"], row["proj_mean_abs_corr"],
            row["proj_effective_rank"],
        )
        rows.append(row)
    total_runtime = time.time() - t_all

    if not rows:
        log.error("No checkpoints evaluated successfully.")
        return 1

    # ---- write outputs ----
    csv_path = out_dir / "encoder_audit.csv"
    md_path = out_dir / "encoder_audit.md"
    _write_csv(rows, csv_path)
    _write_markdown(
        rows, md_path,
        n_samples=n_samples,
        ckpt_name=args.ckpt_name,
        device=str(device),
        total_runtime_s=total_runtime,
    )
    log.info("Wrote %s", csv_path)
    log.info("Wrote %s", md_path)
    log.info("Total runtime: %.1fs", total_runtime)

    # ---- sanity checks ----
    _run_sanity_checks(rows, log)

    return 0


if __name__ == "__main__":
    sys.exit(main())
