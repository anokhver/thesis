"""Real ML pipeline: embedding extraction + Leiden clustering.

Replaces ``services.work_stub.do_work``. Heavy imports (torch, monai,
umap, leidenalg) are deferred inside ``do_work`` so Django startup
stays fast.

Architectural notes:

* Extraction replicates the loop from
  ``scripts/evaluation/extract_embeddings.py`` inline. The package's
  ``synaptic_ssl.clustering.embeddings.extract_patch_embeddings``
  helper does NOT apply ``ValSingleViewTransform`` normalisation, so
  it is not a drop-in. Refactoring the upstream training code is out
  of scope (rule 5).
* Clustering reuses ``scripts/run_clustering.py``'s ``run(args)``
  entry point. The script is not a package so we load it via
  ``importlib.util.spec_from_file_location`` and cache the module.
* Clustering's ``run()`` adds a ``FileHandler`` to the root logger
  without removing it. We snapshot/restore root handlers around each
  call so repeated runs do not accumulate handlers or leave file
  descriptors open on Windows.
* ``torch.load(..., weights_only=False)`` is required because
  training checkpoints carry optimizer state (and ``weights_only=True``
  rejects them). Mitigations live in ``_validate_checkpoint``; the
  trust assumption is "single user, localhost". Add auth before
  exposing this beyond localhost.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import logging
import math
import os
import shutil
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from django.conf import settings
from django.db import transaction

from synapse_web.models import AnalysisRun, SourceImageStats

logger = logging.getLogger(__name__)

_run_clustering_module = None
_RUN_CLUSTERING_SCRIPT = Path(settings.BASE_DIR) / "scripts" / "run_clustering.py"


# ---------------------------------------------------------------------------
# JSON sanitiser
# ---------------------------------------------------------------------------

def _to_json_safe(obj):
    """Return a JSON-encodable copy of ``obj``.

    Handles numpy scalars/arrays, dataclasses, namedtuples, dict/list/tuple,
    and non-finite floats (NaN/Inf) — both numpy and built-in. Django's
    JSONField rejects NaN/Inf, so we replace them with ``None``.
    """
    # Lazy: numpy is only available inside the worker.
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is installed in this app
        np = None

    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if np is not None:
        if isinstance(obj, np.floating):
            f = float(obj)
            return f if math.isfinite(f) else None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return [_to_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if hasattr(obj, "_asdict"):
        return _to_json_safe(obj._asdict())
    if is_dataclass(obj):
        return _to_json_safe(asdict(obj))
    return obj


# ---------------------------------------------------------------------------
# Checkpoint guardrails
# ---------------------------------------------------------------------------

def _validate_checkpoint(path_str: str) -> Path:
    if not path_str:
        raise RuntimeError("Checkpoint path is empty.")
    path = Path(path_str)
    if not path.is_file():
        raise RuntimeError(f"Checkpoint not found: {path}.")
    if path.is_symlink():
        raise RuntimeError(f"Checkpoint is a symlink (refused): {path}.")
    limit = int(getattr(settings, "MAX_CHECKPOINT_UPLOAD_BYTES", 0)) or None
    if limit is not None and path.stat().st_size > limit:
        raise RuntimeError(
            f"Checkpoint is larger than MAX_CHECKPOINT_UPLOAD_BYTES."
        )
    return path


# ---------------------------------------------------------------------------
# Extraction (extract_and_cluster only)
# ---------------------------------------------------------------------------

def _extract_and_write_bundle(
    run: AnalysisRun, checkpoint_path: str, reporter,
) -> None:
    """Run the encoder over the patches at run.input_dir and write a
    run_clustering-compatible bundle into run.bundle_dir."""
    import numpy as np
    import torch
    from synaptic_ssl.training.config import ModelCfg
    from synaptic_ssl.training.augment import ValSingleViewTransform
    from synaptic_ssl.training.data import compute_channel_stats
    from synaptic_ssl.models.swin import build_swin_encoder
    from synaptic_ssl.utils_data.patch_dataset import PatchDataset

    ckpt_path = _validate_checkpoint(checkpoint_path)
    input_dir = Path(run.input_dir)
    bundle_dir = Path(run.bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reporter.update(progress=10, message=f"Loading encoder on {device}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "encoder_state_dict" not in ckpt:
        raise RuntimeError(
            "Checkpoint missing 'encoder_state_dict'. Use a training "
            "checkpoint produced by synaptic_ssl."
        )

    cfg = ModelCfg()
    encoder = build_swin_encoder(cfg).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    ch_mean = ckpt.get("channel_mean")
    ch_std = ckpt.get("channel_std")
    raw = PatchDataset(root=input_dir)
    if ch_mean is None or ch_std is None:
        reporter.update(progress=15, message="Computing channel statistics")
        ch_mean, ch_std = compute_channel_stats(
            raw, in_channels=cfg.in_channels, max_samples=2048,
        )
    ch_mean = torch.as_tensor(ch_mean)
    ch_std = torch.as_tensor(ch_std)
    transform = ValSingleViewTransform(ch_mean, ch_std)

    # num_workers=0 on Windows: DataLoader's fork mode is fragile in
    # a worker thread + multiprocessing combination.
    loader = torch.utils.data.DataLoader(
        raw, batch_size=64, shuffle=False, num_workers=0,
        pin_memory=(device.type != "cpu"),
    )

    feats: list = []
    n_batches = max(len(loader), 1)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch if torch.is_tensor(batch) else batch[0]
            x = transform(x).to(device, non_blocking=True).contiguous()
            z = encoder(x)[-1].mean(dim=(-2, -1))
            feats.append(z.float().cpu().numpy())
            reporter.update(
                progress=20 + int(30 * (i + 1) / n_batches),
                message=f"Extracting embeddings ({i + 1}/{n_batches})",
            )

    Z = np.concatenate(feats, axis=0).astype(np.float32)
    filenames = [r["filename"] for r in raw.records]
    source_images = [
        Path(str(r.get("source_image") or r.get("source_npy")
                 or r.get("source_path") or "UNKNOWN")).name
        for r in raw.records
    ]
    # image_index: use existing column when parseable, otherwise derive.
    image_indices: list[int] = []
    derived: dict[str, int] = {}
    for r, src in zip(raw.records, source_images):
        raw_idx = str(r.get("image_index", "")).strip()
        try:
            image_indices.append(int(raw_idx))
        except ValueError:
            if src not in derived:
                derived[src] = len(derived)
            image_indices.append(derived[src])
    # If everything fell back to derivation, the loop above produced a
    # consistent contiguous mapping; if mixed, downstream tolerates it.

    # Treatment groups from the upload manifest, defaulting to "UNKNOWN".
    tg_map = {
        str(item.get("source_image")): (item.get("treatment_group") or "UNKNOWN")
        for item in (run.input_manifest or [])
    }
    treatment_groups = [tg_map.get(s, "UNKNOWN") for s in source_images]

    reporter.update(progress=55, message="Writing bundle")
    np.save(bundle_dir / "embeddings.npy", Z)
    with open(bundle_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "source_image", "image_index", "treatment_group"])
        for name, src, idx, tg in zip(filenames, source_images, image_indices, treatment_groups):
            w.writerow([name, src, idx, tg])
    with open(bundle_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {"run_id": str(run.id), "checkpoint_path": str(ckpt_path)},
            f, indent=2,
        )

    AnalysisRun.objects.filter(id=run.id).update(
        embedding_dim=int(Z.shape[1]),
        bundle_dir=str(bundle_dir),
    )


# ---------------------------------------------------------------------------
# Clustering (both run kinds)
# ---------------------------------------------------------------------------

def _load_run_clustering():
    """Load scripts/run_clustering.py as a module (cached)."""
    global _run_clustering_module
    if _run_clustering_module is not None:
        return _run_clustering_module
    if not _RUN_CLUSTERING_SCRIPT.is_file():
        raise RuntimeError(
            f"run_clustering.py not found at {_RUN_CLUSTERING_SCRIPT}."
        )
    spec = importlib.util.spec_from_file_location(
        "synapse_web_run_clustering", _RUN_CLUSTERING_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    # The script needs `src/` on sys.path to import synaptic_ssl, but
    # the package is already installed in our venv via pip install -e,
    # so the path manipulation inside the script is a no-op.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _run_clustering_module = module
    return module


def _run_clustering(bundle_dir: Path, output_dir: Path) -> None:
    """Call run_clustering.run(args). Wipes output_dir first.

    Snapshots root logger handlers so the upstream FileHandler that the
    script adds (and never removes) does not leak across runs.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    module = _load_run_clustering()
    args = module.parse_args([
        "--bundle", str(bundle_dir),
        "--output-dir", str(output_dir),
        "--overwrite",
    ])

    root = logging.getLogger()
    pre = list(root.handlers)
    try:
        rc = module.run(args)
    finally:
        # Remove and close anything the upstream call added.
        for h in list(root.handlers):
            if h not in pre:
                root.removeHandler(h)
                try:
                    h.close()
                except Exception:  # noqa: BLE001
                    pass
    if rc != 0:
        raise RuntimeError(f"run_clustering exited with code {rc}.")


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def _dominant_cluster(cluster_counts: dict[str, int]) -> int | None:
    if not cluster_counts:
        return None
    items = [(int(k), int(v)) for k, v in cluster_counts.items()]
    non_noise = [(c, n) for c, n in items if c != -1]
    pool = non_noise if non_noise else items
    if not pool:
        return None
    max_n = max(n for _, n in pool)
    winners = sorted(c for c, n in pool if n == max_n)
    return winners[0]


def _ingest_results(run: AnalysisRun, output_dir: Path) -> None:
    import pandas as pd

    summary_path = output_dir / "summary.json"
    labels_path = output_dir / "patch_labels.csv"
    if not summary_path.is_file():
        raise RuntimeError(f"summary.json not found at {summary_path}.")
    if not labels_path.is_file():
        raise RuntimeError(f"patch_labels.csv not found at {labels_path}.")

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    summary_safe = _to_json_safe(summary)

    data = summary.get("data") or {}
    pick = summary.get("resolution_pick") or {}

    def _as_int(v):
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _as_float(v):
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    n_clusters = _as_int(pick.get("n_clusters") or data.get("n_clusters"))
    embedding_dim = _as_int(data.get("embedding_dim"))
    pca_dim = _as_int(data.get("pca_dim"))
    picked_resolution = _as_float(pick.get("resolution"))
    mean_ari = _as_float(pick.get("mean_ari"))

    df = pd.read_csv(labels_path)
    if "source_image" not in df.columns or "cluster" not in df.columns:
        raise RuntimeError(
            "patch_labels.csv missing required columns (source_image, cluster)."
        )

    tg_map = {
        str(item.get("source_image")): (item.get("treatment_group") or "UNKNOWN")
        for item in (run.input_manifest or [])
    }

    with transaction.atomic():
        for src, group in df.groupby("source_image"):
            counts = group["cluster"].astype(int).value_counts()
            cluster_counts = {str(int(k)): int(v) for k, v in counts.items()}
            SourceImageStats.objects.update_or_create(
                analysis_run=run,
                source_image=str(src),
                defaults={
                    "treatment_group": tg_map.get(str(src), "UNKNOWN"),
                    "n_patches": int(len(group)),
                    "cluster_counts": cluster_counts,
                    "dominant_cluster": _dominant_cluster(cluster_counts),
                },
            )

        AnalysisRun.objects.filter(id=run.id).update(
            clustering_summary=summary_safe,
            n_clusters=n_clusters,
            embedding_dim=embedding_dim if embedding_dim is not None else run.embedding_dim,
            pca_dim=pca_dim,
            picked_resolution=picked_resolution,
            mean_ari=mean_ari,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def do_work(
    *,
    run_id: str,
    inputs: list[dict],
    checkpoint_path: str,
    output_dir: str,
    config: dict,
    reporter,
) -> None:
    run = AnalysisRun.objects.get(id=run_id)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reporter.update(progress=5, message="Preparing")
    if run.run_kind == "extract_and_cluster":
        _extract_and_write_bundle(run, checkpoint_path, reporter)
        bundle_dir = Path(run.bundle_dir)
    else:
        if not run.bundle_dir:
            raise RuntimeError("cluster_only run has no bundle_dir set.")
        bundle_dir = Path(run.bundle_dir)

    reporter.update(progress=60, message="Clustering")
    _run_clustering(bundle_dir, out_dir)

    reporter.update(progress=90, message="Ingesting summary")
    _ingest_results(run, out_dir)
    reporter.update(progress=100, message="Done")
