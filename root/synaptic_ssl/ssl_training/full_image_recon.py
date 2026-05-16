"""Sliding-window full-image SimMIM reconstruction (post-training viz)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from ..training.masking import apply_mask, random_block_mask


@torch.no_grad()
def full_image_sliding_recon(
    encoder: nn.Module,
    heads: dict[str, nn.Module],
    full_image: np.ndarray,
    *,
    patch_size: int,
    mask_block_size: int,
    mask_ratio: float,
    head_stage_index: int,
    ch_mean: torch.Tensor,
    ch_std: torch.Tensor,
    device: torch.device,
    overlap: float = 0.5,
    batch_size: int = 16,
    seed: int = 0,
):
    """Sliding-window SimMIM reconstruction on one ``(C, H, W)`` full image.

    Returns ``(full_recon, full_masked, mask_freq, n_windows)`` cropped back
    to the original ``(H, W)``.
    """
    encoder.eval()
    for h in heads.values():
        h.eval()

    C, H, W = full_image.shape
    stride = max(1, int(patch_size * (1.0 - overlap)))

    ch_mean_np = np.asarray(ch_mean.cpu(), dtype=np.float32).reshape(-1, 1, 1)
    ch_std_np  = np.asarray(ch_std.cpu(),  dtype=np.float32).reshape(-1, 1, 1)

    pad_h = (stride - (H - patch_size) % stride) % stride if H > patch_size else patch_size - H
    pad_w = (stride - (W - patch_size) % stride) % stride if W > patch_size else patch_size - W
    padded = np.pad(full_image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    pH, pW = padded.shape[1], padded.shape[2]

    acc_recon  = np.zeros((C, pH, pW), dtype=np.float64)
    acc_masked = np.zeros((C, pH, pW), dtype=np.float64)
    acc_mask   = np.zeros((pH, pW),    dtype=np.float64)
    count      = np.zeros((pH, pW),    dtype=np.float64)

    positions = [
        (y, x)
        for y in range(0, pH - patch_size + 1, stride)
        for x in range(0, pW - patch_size + 1, stride)
    ]

    torch.manual_seed(seed)
    mu = ch_mean.view(1, -1, 1, 1).to(device)
    sd = ch_std.view(1, -1, 1, 1).to(device)

    for i in tqdm(range(0, len(positions), batch_size), desc="sliding"):
        batch_pos = positions[i : i + batch_size]
        patches = [
            (padded[:, y : y + patch_size, x : x + patch_size] - ch_mean_np) / ch_std_np
            for y, x in batch_pos
        ]
        view = torch.from_numpy(np.stack(patches)).float().to(device)

        mask = random_block_mask(view, mask_block_size, mask_ratio)
        v_masked = apply_mask(view, mask, heads["mask_token"])
        z = encoder(v_masked.contiguous())[head_stage_index]
        recon = heads["decoder"](z).float()

        recon01  = (recon    * sd + mu).clamp(0, 1).cpu().numpy()
        masked01 = (v_masked * sd + mu).clamp(0, 1).cpu().numpy()
        mask_np  = mask.cpu().numpy()[:, 0]

        for j, (y, x) in enumerate(batch_pos):
            acc_recon [:, y : y + patch_size, x : x + patch_size] += recon01[j]
            acc_masked[:, y : y + patch_size, x : x + patch_size] += masked01[j]
            acc_mask  [   y : y + patch_size, x : x + patch_size] += mask_np[j]
            count     [   y : y + patch_size, x : x + patch_size] += 1.0

    cs = np.maximum(count, 1.0)
    full_recon  = (acc_recon  / cs)[:, :H, :W].astype(np.float32)
    full_masked = (acc_masked / cs)[:, :H, :W].astype(np.float32)
    mask_freq   = (acc_mask   / cs)[:H, :W].astype(np.float32)
    return full_recon, full_masked, mask_freq, len(positions)


def run_full_image_recon(
    encoder: nn.Module,
    heads: dict[str, nn.Module],
    data_cfg,
    model_cfg,
    ssl_cfg,
    base_cfg,
    ch_mean: torch.Tensor,
    ch_std: torch.Tensor,
    save_dir: str | Path,
    run_label: str | None,
    device: torch.device,
    full_image_cfg: dict,
    logger,
) -> None:
    """Run full-image sliding-window reconstruction and save .npz + .png."""
    from ..utils_data.reassemble import reassemble_image, list_image_indices

    save_dir = Path(save_dir)

    img_index  = full_image_cfg.get("image_index", None)
    overlap    = full_image_cfg.get("overlap", 0.5)
    batch_size = full_image_cfg.get("batch_size", 16)

    idxs = list_image_indices(data_cfg.data_root, exclude_patterns=data_cfg.exclude_patterns)
    img_idx = idxs[0] if img_index is None else img_index
    full_image, _records = reassemble_image(
        data_cfg.data_root, img_idx,
        exclude_patterns=data_cfg.exclude_patterns,
    )
    logger.info(
        f"full image idx={img_idx}  shape={tuple(full_image.shape)}  "
        f"source={_records[0]['source_image']}"
    )

    full_recon, full_masked, mask_freq, n_windows = full_image_sliding_recon(
        encoder, heads, full_image,
        patch_size=model_cfg.img_size,
        mask_block_size=ssl_cfg.mask_block_size,
        mask_ratio=ssl_cfg.mask_ratio,
        head_stage_index=ssl_cfg.head_stage_index,
        ch_mean=ch_mean, ch_std=ch_std,
        device=device,
        overlap=overlap,
        batch_size=batch_size,
        seed=base_cfg.seed + 31,
    )

    err = np.abs(full_recon - full_image).mean(0)
    stride_px = int(model_cfg.img_size * (1 - overlap))
    logger.info(
        f"full-image recon: windows={n_windows}  stride={stride_px}px  "
        f"mean|err|={float(err.mean()):.4f}  max|err|={float(err.max()):.4f}"
    )

    np.savez_compressed(
        save_dir / "post_recon_fullimage.npz",
        image_index=np.array(img_idx),
        full_image=full_image,
        full_recon=full_recon,
        full_masked=full_masked,
        mask_freq=mask_freq,
        overlap=np.float32(overlap),
    )

    C, H, W = full_image.shape
    ncols = max(3, C)
    fig, axes = plt.subplots(4, ncols, figsize=(4 * ncols, 16), squeeze=False)
    for r in range(4):
        for c in range(ncols):
            axes[r, c].axis("off")

    for ci, name in enumerate(data_cfg.channel_names):
        axes[0, ci].imshow(full_image[ci], cmap="magma", vmin=0, vmax=1)
        axes[0, ci].set_title(f"target  {name}")
        axes[1, ci].imshow(full_masked[ci], cmap="magma", vmin=0, vmax=1)
        axes[1, ci].set_title(f"masked input (avg)  {name}")
        axes[2, ci].imshow(full_recon[ci], cmap="magma", vmin=0, vmax=1)
        axes[2, ci].set_title(f"reconstruction  {name}")

    axes[3, 0].imshow(np.transpose(np.clip(full_image, 0, 1), (1, 2, 0)))
    axes[3, 0].set_title("target  RGB")
    axes[3, 1].imshow(np.transpose(np.clip(full_recon, 0, 1), (1, 2, 0)))
    axes[3, 1].set_title("reconstruction  RGB")
    err_vmax = max(float(err.max()), 1e-6)
    err_im = axes[3, 2].imshow(err, cmap="hot", vmin=0, vmax=err_vmax)
    axes[3, 2].set_title(f"|err|  mean={float(err.mean()):.3f}")
    fig.colorbar(err_im, ax=axes[3, 2], fraction=0.046, pad=0.02)

    mask_pct = int(ssl_cfg.mask_ratio * 100)
    fig.suptitle(
        f"POST: full-image sliding-window reconstruction  "
        f"(img={img_idx}, stride={stride_px}px, "
        f"mask={mask_pct}% blk={ssl_cfg.mask_block_size}, "
        f"windows={n_windows})"
    )
    if run_label:
        fig.text(
            0.01, 0.985, run_label, ha="left", va="top", fontsize=8,
            color="#444", family="DejaVu Sans Mono",
        )
    fig.tight_layout(rect=(0, 0, 0.97, 0.97))
    fig.savefig(save_dir / "post_recon_fullimage.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
