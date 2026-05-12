"""Plot training visualisations.

Produce Matplotlib figures from tensors/arrays. Never call ``plt.show()``.
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


# --------------------------------------------------------------------- labels

# Mathtext labels for every metric key emitted by ``losses.py`` and the CSV
# logger. Used by every curve plot so figures speak the same language as the
# thesis text.
LOSS_LABELS: dict[str, str] = {
    "loss":          r"$\mathcal{L}_\mathrm{total}$",
    "ssl_loss":      r"$\mathcal{L}_\mathrm{total}$",
    "train_loss":    r"$\mathcal{L}_\mathrm{total}$ (train)",
    "val_loss":      r"$\mathcal{L}_\mathrm{total}$ (val)",
    "val_metric":    r"val metric ($\mathcal{L}_\mathrm{recon}$)",
    "recon":         r"$\mathcal{L}_\mathrm{recon}$ (SimMIM)",
    "train_recon":   r"$\mathcal{L}_\mathrm{recon}$ (train)",
    "val_recon":     r"$\mathcal{L}_\mathrm{recon}$ (val)",
    "sim":           r"$\mathcal{L}_\mathrm{sim}$ (VICReg invariance)",
    "train_sim":     r"$\mathcal{L}_\mathrm{sim}$ (train)",
    "std":           r"$\mathcal{L}_\mathrm{var}$ (VICReg variance)",
    "train_std":     r"$\mathcal{L}_\mathrm{var}$ (train)",
    "val_std":       r"$\mathcal{L}_\mathrm{var}$ (val)",
    "cov":           r"$\mathcal{L}_\mathrm{cov}$ (VICReg covariance)",
    "train_cov":     r"$\mathcal{L}_\mathrm{cov}$ (train)",
    "val_cov":       r"$\mathcal{L}_\mathrm{cov}$ (val)",
    "vicreg":        r"$\mathcal{L}_\mathrm{VICReg}$",
    "train_vicreg":  r"$\mathcal{L}_\mathrm{VICReg}$ (train)",
    "lr_encoder":    r"lr (encoder)",
    "lr_head":       r"lr (heads)",
}


def loss_label(key: str) -> str:
    """Return the math-style label for ``key`` (falls back to the key itself)."""
    return LOSS_LABELS.get(key, key)


def _save_or_show(fig, save_to: str | Path | None):
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=120, bbox_inches="tight")


def _annotate_run(fig, run_label: str | None):
    """Print a small left-aligned identifier (method / init / epoch) below the suptitle."""
    if not run_label:
        return
    fig.text(
        0.01, 0.985, run_label,
        ha="left", va="top", fontsize=8, color="#444",
        family="DejaVu Sans Mono",
    )


def _to_disp(t: torch.Tensor) -> np.ndarray:
    """Convert ``(C, H, W)`` to a display array.

    Return ``(H, W, 3)`` for 3 channels. Return channel mean otherwise.
    """
    if t.shape[0] == 3:
        return t.permute(1, 2, 0).cpu().numpy()
    return t.mean(0).cpu().numpy()


def plot_two_views(
    raw_subset,
    transform,
    indices: Sequence[int],
    channel_names: Sequence[str],
    *,
    suptitle: str = "Two-view augmentation",
    save_to: str | Path | None = None,
):
    """Plot ``view1 / view2`` of every channel for the given indices."""
    n = len(indices)
    C = len(channel_names)
    fig, axes = plt.subplots(n, 2 * C, figsize=(2 * 2 * C, 2 * n), squeeze=False)
    for r, idx in enumerate(indices):
        a, b = transform(raw_subset[idx])
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        for ci, name in enumerate(channel_names):
            axes[r, ci].imshow(a[ci], cmap="magma")
            axes[r, ci].set_title(f"v1 {name}" if r == 0 else "")
            axes[r, ci].axis("off")
            axes[r, C + ci].imshow(b[ci], cmap="magma")
            axes[r, C + ci].set_title(f"v2 {name}" if r == 0 else "")
            axes[r, C + ci].axis("off")
    fig.suptitle(suptitle)
    fig.tight_layout()
    _save_or_show(fig, save_to)
    return fig


def plot_channel_histograms(
    batch: torch.Tensor,
    channel_names: Sequence[str],
    *,
    suptitle: str = "Post-normalisation channel histograms",
    save_to: str | Path | None = None,
):
    """Plot a histogram of each channel after normalisation."""
    C = len(channel_names)
    fig, axes = plt.subplots(1, C, figsize=(4 * C, 3), squeeze=False)
    axes = axes[0]
    for ci, name in enumerate(channel_names):
        pix = batch[:, ci].flatten().cpu().numpy()
        axes[ci].hist(pix, bins=80, color="steelblue", alpha=0.85)
        axes[ci].set_title(f"{name}\nmean={pix.mean():.2f}  std={pix.std():.2f}")
        axes[ci].axvline(0, color="k", linestyle="--", linewidth=0.8)
    fig.suptitle(suptitle)
    fig.tight_layout()
    _save_or_show(fig, save_to)
    return fig


def plot_recon_panel(
    view: torch.Tensor,
    masked_view: torch.Tensor,
    recon: torch.Tensor,
    mask: torch.Tensor,
    indices: Sequence[int],
    *,
    ch_mean: torch.Tensor,
    ch_std: torch.Tensor,
    suptitle: str = "Reconstruction",
    save_to: str | Path | None = None,
    run_label: str | None = None,
    channel_names: Sequence[str] | None = None,
):
    """Plot 4 rows x N cols: target | masked input | reconstruction | absolute error.

    Share one colorbar across the error row. Show per-channel mean error when
    ``channel_names`` is supplied.
    """
    n = view.shape[0]
    mu = ch_mean.view(1, -1, 1, 1).to(view.device)
    sd = ch_std.view(1, -1, 1, 1).to(view.device)

    inp = (view * sd + mu).cpu()
    rec = (recon * sd + mu).cpu()
    msk = (masked_view * sd + mu).cpu()
    mask_np = mask.cpu().squeeze(1).numpy()
    err_per_ch = (rec - inp).abs()             # (B, C, H, W) -- per-channel error
    err = err_per_ch.mean(dim=1)               # (B, H, W)

    inp_disp = inp.clamp(0, 1)
    rec_disp = rec.clamp(0, 1)
    msk_disp = msk.clamp(0, 1)

    err_vmax = float(err.max().item()) if err.numel() else 1.0
    err_vmax = max(err_vmax, 1e-6)

    fig, axes = plt.subplots(4, n, figsize=(3.2 * n, 12.0), squeeze=False)
    err_img = None
    for j in range(n):
        axes[0, j].imshow(_to_disp(inp_disp[j]))
        axes[0, j].set_title(f"target  idx={indices[j]}")
        axes[0, j].axis("off")
        axes[1, j].imshow(_to_disp(msk_disp[j]))
        axes[1, j].contour(mask_np[j], levels=[0.5], colors=["c"], linewidths=0.7)
        axes[1, j].set_title(f"masked input  ({mask_np[j].mean() * 100:.0f}% hidden)")
        axes[1, j].axis("off")
        axes[2, j].imshow(_to_disp(rec_disp[j]))
        axes[2, j].set_title("reconstruction")
        axes[2, j].axis("off")
        err_img = axes[3, j].imshow(err[j].numpy(), cmap="hot", vmin=0, vmax=err_vmax)
        if channel_names is not None and err_per_ch.shape[1] == len(channel_names):
            per = err_per_ch[j].mean(dim=(-2, -1)).tolist()
            tag = "\n".join(f"{n_}={v:.2f}" for n_, v in zip(channel_names, per))
            axes[3, j].set_title(f"|err|  mean={err[j].mean().item():.3f}", fontsize=9)
            axes[3, j].set_xlabel(tag, fontsize=7, labelpad=2)
        else:
            axes[3, j].set_title(f"|err|  mean={err[j].mean().item():.3f}")
        axes[3, j].axis("off")

    fig.suptitle(suptitle)
    _annotate_run(fig, run_label)
    fig.tight_layout(rect=(0, 0.02, 0.92, 0.95))

    if err_img is not None:
        cax = fig.add_axes([0.93, 0.05, 0.015, 0.20])
        cbar = fig.colorbar(err_img, cax=cax)
        cbar.ax.set_ylabel("|err|  (data units)", fontsize=9)
    _save_or_show(fig, save_to)
    return fig


def plot_loss_curves(
    csv_path: str | Path,
    *,
    train_key: str = "train_loss",
    val_key: str = "val_metric",
    suptitle: str = "Training curves",
    save_to: str | Path | None = None,
    log_y: bool = True,
    run_label: str | None = None,
):
    """Plot training curves from the CSV produced by ``CSVMetricLogger``.

    2x2 grid: total objective, SimMIM recon, VICReg components, LR schedule.
    Falls back gracefully when columns are missing.
    """
    rows = list(_csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        return None

    def col(name: str) -> tuple[list[int], list[float]]:
        xs, ys = [], []
        for r in rows:
            v = r.get(name)
            if v in ("", None):
                continue
            try:
                ys.append(float(v))
                xs.append(int(r["epoch"]))
            except (ValueError, KeyError):
                continue
        return xs, ys

    has = lambda k: any(r.get(k) not in ("", None) for r in rows)

    # Pick best epoch from CSV if available; otherwise compute from train_key.
    best_ep, best_val = None, None
    if has("best_epoch") and has("best_val_metric"):
        last = rows[-1]
        try:
            best_ep = int(last["best_epoch"])
            best_val = float(last["best_val_metric"])
        except (ValueError, KeyError):
            best_ep, best_val = None, None
    if best_ep is None:
        x_v, y_v = col(val_key)
        if y_v:
            i = int(np.argmin(y_v))
            best_ep, best_val = x_v[i], y_v[i]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax_total, ax_recon = axes[0, 0], axes[0, 1]
    ax_vic,   ax_lr    = axes[1, 0], axes[1, 1]

    # --- (0,0) total objective
    for k in (train_key, val_key):
        x, y = col(k)
        if y:
            ax_total.plot(x, y, label=loss_label(k), linewidth=1.4)
    if best_ep is not None and best_val is not None:
        ax_total.axvline(best_ep, color="k", linestyle=":", linewidth=0.9, alpha=0.7)
        ax_total.scatter([best_ep], [best_val], s=22, color="k", zorder=5)
        ax_total.annotate(
            f"best @ epoch {best_ep}\n{val_key}={best_val:.4f}",
            xy=(best_ep, best_val),
            xytext=(8, 8), textcoords="offset points",
            fontsize=8, color="k",
        )
    ax_total.set_xlabel("epoch")
    ax_total.set_ylabel("loss")
    ax_total.set_title("Total objective")
    if log_y:
        ax_total.set_yscale("log")
    ax_total.legend(fontsize=9)
    ax_total.grid(alpha=0.25)

    # --- (0,1) SimMIM recon
    plotted = False
    for k in ("train_recon", "val_recon"):
        if has(k):
            x, y = col(k)
            ax_recon.plot(x, y, label=loss_label(k), linewidth=1.4)
            plotted = True
    if plotted:
        ax_recon.set_xlabel("epoch")
        ax_recon.set_ylabel(r"$\mathcal{L}_\mathrm{recon}$")
        ax_recon.set_title("SimMIM masked reconstruction")
        if log_y:
            ax_recon.set_yscale("log")
        ax_recon.legend(fontsize=9)
        ax_recon.grid(alpha=0.25)
    else:
        ax_recon.set_visible(False)

    # --- (1,0) VICReg components
    plotted = False
    for k in ("train_sim", "train_std", "train_cov", "train_vicreg"):
        if has(k):
            x, y = col(k)
            ax_vic.plot(x, y, label=loss_label(k), linewidth=1.2)
            plotted = True
    if plotted:
        ax_vic.set_xlabel("epoch")
        ax_vic.set_ylabel("loss term")
        ax_vic.set_title("VICReg components (train)")
        if log_y:
            ax_vic.set_yscale("log")
        ax_vic.legend(fontsize=8)
        ax_vic.grid(alpha=0.25)
    else:
        ax_vic.set_visible(False)

    # --- (1,1) LR schedule
    plotted = False
    for k in ("lr_encoder", "lr_head"):
        if has(k):
            x, y = col(k)
            ax_lr.plot(x, y, label=loss_label(k), linewidth=1.4)
            plotted = True
    if plotted:
        ax_lr.set_xlabel("epoch")
        ax_lr.set_ylabel("learning rate")
        ax_lr.set_title("LR schedule")
        ax_lr.set_yscale("log")
        ax_lr.legend(fontsize=9)
        ax_lr.grid(alpha=0.25)
    else:
        ax_lr.set_visible(False)

    fig.suptitle(suptitle)
    _annotate_run(fig, run_label)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_or_show(fig, save_to)
    return fig


def plot_overfit_curves(
    history: dict[str, list[float]],
    *,
    suptitle: str = "Overfit-on-batch",
    save_to: str | Path | None = None,
    run_label: str | None = None,
):
    """Plot two log-scale panels for the overfit-on-batch sanity check.

    Left: total objective + SimMIM recon. Right: VICReg sim/std/cov.
    """
    eps = 1e-12

    def _plot(ax, keys, ylabel, title):
        plotted = False
        for k in keys:
            vs = history.get(k)
            if not vs or not any(v > eps for v in vs):
                continue
            ax.plot([max(v, eps) for v in vs], label=loss_label(k), linewidth=1.4)
            plotted = True
        if plotted:
            ax.set_yscale("log")
            ax.set_xlabel("step")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(alpha=0.25)
        return plotted

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    left_ok  = _plot(axes[0], ["loss", "recon"],         "loss (log)",  "Total objective and SimMIM recon")
    right_ok = _plot(axes[1], ["sim", "std", "cov"],     "term (log)",  "VICReg components")
    if not left_ok:
        axes[0].set_visible(False)
    if not right_ok:
        axes[1].set_visible(False)
    fig.suptitle(suptitle)
    _annotate_run(fig, run_label)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_or_show(fig, save_to)
    return fig


def plot_embedding_2d(
    Z2: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    title: str = "Embedding (2D)",
    save_to: str | Path | None = None,
    singular_values: np.ndarray | None = None,
    method: str | None = None,
    effective_rank_value: float | None = None,
    mean_pairwise_cos_value: float | None = None,
    n_samples: int | None = None,
    run_label: str | None = None,
):
    """Scatter ``Z2`` with optional singular-value spectrum panel.

    Stamps N, effective rank, and mean pairwise cosine on the figure.
    When ``method='pca'``, axis labels show explained-variance fractions.
    """
    has_spec = singular_values is not None
    fig, axes = plt.subplots(1, 2 if has_spec else 1,
                             figsize=(12 if has_spec else 6, 5),
                             squeeze=False)
    ax = axes[0, 0]
    if labels is None:
        ax.scatter(Z2[:, 0], Z2[:, 1], s=5, alpha=0.6)
    else:
        sc = ax.scatter(Z2[:, 0], Z2[:, 1], s=5, alpha=0.7, c=labels, cmap="tab20")
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title(title)
    if method and method.lower() == "pca" and has_spec:
        s = np.asarray(singular_values, dtype=np.float64)
        var = s ** 2
        if len(var) >= 2 and var.sum() > 0:
            v1, v2 = var[:2] / var.sum()
            ax.set_xlabel(f"PC 1 ({v1 * 100:.1f}% var)")
            ax.set_ylabel(f"PC 2 ({v2 * 100:.1f}% var)")
        else:
            ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    else:
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")

    # Stamp collapse diagnostics in the upper-left corner of the scatter.
    diag_lines = []
    if n_samples is not None:
        diag_lines.append(f"N = {n_samples}")
    if effective_rank_value is not None:
        if has_spec:
            diag_lines.append(
                f"effective rank = {effective_rank_value:.1f} / "
                f"{len(np.asarray(singular_values))}"
            )
        else:
            diag_lines.append(f"effective rank = {effective_rank_value:.1f}")
    if mean_pairwise_cos_value is not None:
        verdict = "COLLAPSED" if mean_pairwise_cos_value > 0.95 else "OK"
        diag_lines.append(
            r"mean pairwise $\cos$ = "
            f"{mean_pairwise_cos_value:.3f}  ({verdict})"
        )
    if diag_lines:
        ax.text(
            0.02, 0.98, "\n".join(diag_lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#bbb", alpha=0.85),
        )

    if has_spec:
        ax_s = axes[0, 1]
        s = np.asarray(singular_values, dtype=np.float64)
        ax_s.plot(s / max(s.max(), 1e-12), linewidth=1.5)
        ax_s.set_yscale("log")
        ax_s.set_title("Singular value spectrum (normalised)")
        ax_s.set_xlabel("rank")
        ax_s.set_ylabel(r"$\sigma_i / \sigma_1$")
        ax_s.grid(alpha=0.25)
        if effective_rank_value is not None:
            ax_s.axvline(effective_rank_value, color="r", linestyle=":",
                         linewidth=0.9, alpha=0.7,
                         label=f"effective rank = {effective_rank_value:.1f}")
            ax_s.legend(fontsize=8, loc="upper right")
    _annotate_run(fig, run_label)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_or_show(fig, save_to)
    return fig
