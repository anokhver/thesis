"""Iter-1 pseudo-label refresh for the puncta self-training loop.

For every patch in ``patch_dataset``, run the teacher model (S0) under
8-way D4 TTA, threshold the 2-channel probability map at ``threshold``,
keep only connected components whose area falls inside the per-channel
``[A_min, A_max]`` biological band (Harris & Stevens 1989), gate by
centre-on ``dilate(soma U dend, near_dilate_px)``, and atomically
overwrite ``_pre.npy`` / ``_post.npy`` on disk.

Defaults match :mod:`synaptic_ssl.pseudolabels.puncta_log`:

  * PRE  area: ``[20, 75]`` px^2
  * POST area: ``[6, 22]``  px^2
  * near_dilate_px: 4 (DEFAULT_NEAR_DILATE_PX)

The teacher model is expected to be a SwinUNETR with ``out_channels=2``
in :data:`dataset_puncta.PRE_CHANNEL` / :data:`dataset_puncta.POST_CHANNEL`
order. The student dataset built on the refreshed files in the next
iter-1 :func:`runner.run_seg_iter` call will pick the new masks up
through the usual ``_pre.npy`` / ``_post.npy`` lookups.

Soma + dendrite masks are read but *not* modified: they come from the
structural channel pseudo-labellers and are stable across iterations.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..utils_data.patch_dataset import PatchDataset
from .dataset_puncta import POST_CHANNEL, PRE_CHANNEL
from .inference import predict_d4_tta


# Defaults pulled from puncta_log.py {DEFAULT_PUNCTA_CFG_PRE, DEFAULT_PUNCTA_CFG_POST}.
# Citation in their module: Harris & Stevens 1989, J Neurosci 9(8):2982-2997;
# disk area = pi * (sqrt(2) * sigma) ** 2 at sigma in [sigma_min, sigma_max].
PRE_AREA_RANGE_DEFAULT: tuple[int, int] = (20, 75)
POST_AREA_RANGE_DEFAULT: tuple[int, int] = (6, 22)
NEAR_DILATE_PX_DEFAULT: int = 4


@dataclass
class RefreshStats:
    n_patches: int = 0
    n_pre_cc_kept: int = 0
    n_post_cc_kept: int = 0
    n_pre_cc_dropped_area: int = 0
    n_post_cc_dropped_area: int = 0
    n_pre_cc_dropped_gate: int = 0
    n_post_cc_dropped_gate: int = 0
    pre_positive_fraction_sum: float = 0.0
    post_positive_fraction_sum: float = 0.0
    skipped_missing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        n = max(1, self.n_patches)
        return {
            "n_patches":                 self.n_patches,
            "n_pre_cc_kept":             self.n_pre_cc_kept,
            "n_post_cc_kept":            self.n_post_cc_kept,
            "n_pre_cc_dropped_area":     self.n_pre_cc_dropped_area,
            "n_post_cc_dropped_area":    self.n_post_cc_dropped_area,
            "n_pre_cc_dropped_gate":     self.n_pre_cc_dropped_gate,
            "n_post_cc_dropped_gate":    self.n_post_cc_dropped_gate,
            "pre_positive_fraction":     self.pre_positive_fraction_sum / n,
            "post_positive_fraction":    self.post_positive_fraction_sum / n,
            "n_skipped_missing":         len(self.skipped_missing),
        }


def _normalise(patch: np.ndarray, ch_mean: np.ndarray, ch_std: np.ndarray) -> np.ndarray:
    return (patch - ch_mean) / ch_std


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    """Write ``arr`` to ``path`` via tmpfile + rename so partial writes can't corrupt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # NB: ``np.save(str_path, arr)`` silently appends ``.npy`` if the path
    # does not end in it, which would leave the real file at ``tmp + '.npy'``
    # and ``os.replace`` would then move the empty placeholder. Pass an open
    # file handle to bypass that filename rewriting.
    fd, tmp = tempfile.mkstemp(prefix=path.stem + "_", suffix=".npy.tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, arr, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _filter_components(
    prob_channel: np.ndarray,
    *,
    threshold: float,
    area_range: tuple[int, int],
    gate: np.ndarray,
) -> tuple[np.ndarray, int, int, int]:
    """Threshold, area-filter, centre-gate one probability channel.

    A connected component is kept iff
      * ``area in [a_min, a_max]`` (biological band), and
      * the integer centroid of the CC lies inside ``gate``
        (``dilate(soma U dend, near_dilate_px)``).

    Returns ``(kept_mask uint8 (H,W), n_kept, n_dropped_area, n_dropped_gate)``.
    """
    from skimage.measure import label, regionprops

    a_min, a_max = area_range
    binary = (prob_channel >= threshold)
    if not binary.any():
        return np.zeros_like(binary, dtype=np.uint8), 0, 0, 0

    lbl = label(binary, connectivity=2)
    kept = np.zeros_like(binary, dtype=bool)
    n_kept = n_drop_area = n_drop_gate = 0
    gate_h, gate_w = gate.shape

    for prop in regionprops(lbl):
        if prop.area < a_min or prop.area > a_max:
            n_drop_area += 1
            continue
        cr, cc = prop.centroid
        ir = int(round(cr))
        ic = int(round(cc))
        if ir < 0 or ic < 0 or ir >= gate_h or ic >= gate_w or not gate[ir, ic]:
            n_drop_gate += 1
            continue
        kept |= (lbl == prop.label)
        n_kept += 1

    return kept.astype(np.uint8), n_kept, n_drop_area, n_drop_gate


def refresh_pseudolabels(
    model: nn.Module,
    patch_dataset: PatchDataset,
    *,
    pre_dir: str | Path,
    post_dir: str | Path,
    soma_dir: str | Path,
    dend_dir: str | Path,
    ch_mean: torch.Tensor,
    ch_std: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
    pre_area_range: tuple[int, int] = PRE_AREA_RANGE_DEFAULT,
    post_area_range: tuple[int, int] = POST_AREA_RANGE_DEFAULT,
    near_dilate_px: int = NEAR_DILATE_PX_DEFAULT,
    use_tta: bool = True,
    batch_size: int = 16,
    pre_suffix: str = "_pre.npy",
    post_suffix: str = "_post.npy",
    soma_suffix: str = "_soma.npy",
    dend_suffix: str = "_dend.npy",
    overwrite: bool = True,
    progress: bool = True,
    logger=None,
) -> dict:
    """Refresh PRE/POST pseudo-labels under the teacher model.

    Inference is batched (size ``batch_size``); post-processing
    (threshold -> CC area filter -> soma+dend gate -> atomic write) is
    per-patch. The model's normalisation (``ch_mean`` / ``ch_std``) must
    match the training-time normalisation used to produce ``model``
    (typically read from the iter-0 checkpoint).
    """
    from scipy.ndimage import binary_dilation
    from skimage.morphology import disk as _morph_disk

    log = logger.info if logger else (lambda *a, **k: None)
    pre_dir = Path(pre_dir)
    post_dir = Path(post_dir)
    soma_dir = Path(soma_dir)
    dend_dir = Path(dend_dir)
    pre_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    # Guard against silent miscalibration: wrong normalisation stats
    # produce predictions that look reasonable but are systematically
    # off. Force the caller to commit to per-channel finite, positive
    # std with a length matching the patch channel count.
    ch_mean_t = ch_mean.detach().to("cpu").float().view(-1)
    ch_std_t = ch_std.detach().to("cpu").float().view(-1)
    if ch_mean_t.numel() != ch_std_t.numel():
        raise ValueError(
            f"ch_mean ({ch_mean_t.numel()}) and ch_std ({ch_std_t.numel()}) "
            f"must have the same length"
        )
    if not torch.isfinite(ch_std_t).all() or (ch_std_t <= 1e-8).any():
        raise ValueError(
            f"ch_std must be finite and > 1e-8 per channel; got {ch_std_t.tolist()}"
        )
    if not torch.isfinite(ch_mean_t).all():
        raise ValueError(
            f"ch_mean must be finite per channel; got {ch_mean_t.tolist()}"
        )

    struct = _morph_disk(near_dilate_px) if near_dilate_px > 0 else None
    ch_mean_np = ch_mean_t.view(-1, 1, 1).numpy()
    ch_std_np = ch_std_t.view(-1, 1, 1).numpy()
    expected_channels = ch_mean_t.numel()

    if progress:
        from tqdm.auto import tqdm

        record_iter = tqdm(
            range(len(patch_dataset)),
            desc=f"refresh pseudolabels (BS={batch_size}, tta={use_tta})",
        )
    else:
        record_iter = range(len(patch_dataset))

    stats = RefreshStats()
    model.eval()

    buf_records: list[dict] = []
    buf_patches: list[np.ndarray] = []
    buf_gates:   list[np.ndarray] = []

    def _flush_buffer():
        if not buf_records:
            return
        batch = np.stack(buf_patches, axis=0)  # (B, C, H, W)
        batch_t = torch.from_numpy(batch).float().to(device)
        with torch.no_grad():
            if use_tta:
                probs = predict_d4_tta(model, batch_t)  # (B, 2, H, W) for 2-ch head
            else:
                probs = torch.sigmoid(model(batch_t))
        probs_np = probs.detach().cpu().numpy()
        if probs_np.shape[1] != 2:
            raise RuntimeError(
                f"refresh expects a 2-channel teacher (PRE, POST); "
                f"got probs with shape {probs_np.shape}"
            )

        for rec, gate, prob in zip(buf_records, buf_gates, probs_np):
            pre_keep, n_pk, n_pa, n_pg = _filter_components(
                prob[PRE_CHANNEL], threshold=threshold,
                area_range=pre_area_range, gate=gate,
            )
            post_keep, n_qk, n_qa, n_qg = _filter_components(
                prob[POST_CHANNEL], threshold=threshold,
                area_range=post_area_range, gate=gate,
            )

            stem = Path(rec["filename"]).stem
            pre_out = pre_dir / f"{stem}{pre_suffix}"
            post_out = post_dir / f"{stem}{post_suffix}"
            if not overwrite and pre_out.exists() and post_out.exists():
                continue
            _atomic_save(pre_out, pre_keep)
            _atomic_save(post_out, post_keep)

            stats.n_patches += 1
            stats.n_pre_cc_kept += n_pk
            stats.n_post_cc_kept += n_qk
            stats.n_pre_cc_dropped_area += n_pa
            stats.n_post_cc_dropped_area += n_qa
            stats.n_pre_cc_dropped_gate += n_pg
            stats.n_post_cc_dropped_gate += n_qg
            stats.pre_positive_fraction_sum += float(pre_keep.mean())
            stats.post_positive_fraction_sum += float(post_keep.mean())

        buf_records.clear()
        buf_patches.clear()
        buf_gates.clear()

    for i in record_iter:
        rec = patch_dataset.records[i]
        name = rec["filename"]
        stem = Path(name).stem

        soma_p = soma_dir / f"{stem}{soma_suffix}"
        dend_p = dend_dir / f"{stem}{dend_suffix}"
        if not soma_p.exists() or not dend_p.exists():
            stats.skipped_missing.append(name)
            continue

        patch = np.load(patch_dataset.root / name)
        if patch_dataset.channels is not None:
            patch = patch[patch_dataset.channels]
        patch = patch.astype(np.float32, copy=False)
        if patch.shape[0] != expected_channels:
            raise ValueError(
                f"patch {name!r} has {patch.shape[0]} channels but ch_mean/std "
                f"have {expected_channels}; either pass `channels=...` to the "
                f"PatchDataset or update ch_mean/std"
            )

        soma = np.load(soma_p).astype(bool)
        dend = np.load(dend_p).astype(bool)
        if soma.shape != patch.shape[1:] or dend.shape != patch.shape[1:]:
            stats.skipped_missing.append(name)
            continue
        gate = soma | dend
        if struct is not None:
            gate = binary_dilation(gate, structure=struct)

        buf_records.append(rec)
        buf_patches.append(_normalise(patch, ch_mean_np, ch_std_np))
        buf_gates.append(gate)

        if len(buf_records) >= batch_size:
            _flush_buffer()

    _flush_buffer()

    summary = stats.as_dict()
    log(f"[refresh] {summary}")
    return summary
