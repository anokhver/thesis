"""Thesis figure: SimMIM + VICReg pretraining pipeline (SVG only).

Single-figure publication schematic of the joint SimMIM + VICReg + Fourier
self-supervised pretraining pipeline used in ``synaptic_ssl``.

Design goals:
    * No concrete hyper-parameter values (channel counts, depths, mask ratio,
      projector widths, loss weights, ...). The figure must stay valid across
      runs where only the JSON config changes.
    * Computer-Modern math typography for loss symbols (``L_recon`` etc.)
      via matplotlib mathtext, matching standard thesis aesthetics.
    * Two clearly labelled branches leaving a single shared encoder:
          z_clean  ->  GAP  ->  Projector   ->  L_sim / L_std / L_cov
          z_mask   ->  SimMIM decoder       ->  L_recon  +  L_fourier

Usage::

    python imaging/draw_sim_vicreg_architecture.py \
        --out data/imaging_outputs/architecture/pretrain_sim_vicreg.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.ndimage import gaussian_filter


# ── Typography ────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "STIXGeneral", "serif"],
    "mathtext.fontset":   "cm",
    "axes.unicode_minus": False,
})


# ── Palette ───────────────────────────────────────────────────────────
PALETTE = {
    "ink":             "#1F2A37",
    "muted":           "#6B7280",
    "rule":            "#D1D5DB",
    "bg":              "#FAFBFD",
    "input":           "#E8F1FE",  "input_e":        "#4F8BD9",
    "aug":             "#FFF4E0",  "aug_e":          "#E0A24A",
    "view":            "#FFFFFF",  "view_e":         "#E0A24A",
    "mask":            "#F4E5FF",  "mask_e":         "#9B5BD9",
    "encoder":         "#E2F2EC",  "encoder_e":      "#3F9C7A",
    "feat":            "#FFFFFF",  "feat_e":         "#3F9C7A",
    "gap":             "#FFFFFF",  "gap_e":          "#5867D9",
    "proj":            "#E8ECFB",  "proj_e":         "#5867D9",
    "decoder":         "#FDE8EC",  "decoder_e":      "#D24A66",
    "loss_vic":        "#E6E8FA",  "loss_vic_e":     "#5867D9",
    "loss_recon":      "#FDE8EC",  "loss_recon_e":   "#D24A66",
    "loss_fourier":    "#DCE8F4",  "loss_fourier_e": "#3F7FBF",
}


# ── Synthetic microscopy thumbnail ────────────────────────────────────
def _synthetic_microscopy_patch(size: int = 96, seed: int = 7) -> np.ndarray:
    """Generate a tiny illustrative RGB blob pattern (no real data)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        for _ in range(rng.integers(6, 10)):
            cy = int(rng.integers(0, size))
            cx = int(rng.integers(0, size))
            r  = float(rng.uniform(3, 10))
            yy, xx = np.ogrid[:size, :size]
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
            img[..., c] += rng.uniform(0.5, 1.0) * blob
    img = gaussian_filter(img, sigma=(1.2, 1.2, 0.0))
    img -= img.min()
    img /= (img.max() + 1e-8)
    # darken background for an EM-fluorescence look
    img = img ** 1.4
    return img


# ── Primitives ────────────────────────────────────────────────────────
def _rounded(ax, x, y, w, h, *, face, edge, lw=1.4, radius=0.10, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)


def _stack(ax, x, y, w, h, *, face, edge, layers=3, offset=0.10):
    """Stacked-box motif to suggest a multi-stage encoder."""
    for i in range(layers - 1, -1, -1):
        dx = -offset * i
        dy =  offset * i
        layer_face = face if i == 0 else "none"
        _rounded(ax, x + dx, y + dy, w, h,
                 face=layer_face, edge=edge, lw=1.3, zorder=2 + (layers - i))


def _label(ax, x, y, text, *, fs=10, weight="normal", color=None,
           ha="center", va="center", style="normal", zorder=5):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, fontweight=weight,
            color=color or PALETTE["ink"], style=style, zorder=zorder)


def _box(ax, x, y, w, h, *, title, face, edge, sub=None, title_fs=10.5,
         sub_fs=8.4, sub_color=None, title_weight="bold"):
    _rounded(ax, x, y, w, h, face=face, edge=edge)
    if sub is None:
        _label(ax, x + w / 2, y + h / 2, title,
               fs=title_fs, weight=title_weight)
        return
    if isinstance(sub, str):
        sub = [sub]
    # Title sits in the upper part of the box; sub-lines stack below it.
    _label(ax, x + w / 2, y + h * 0.74, title,
           fs=title_fs, weight=title_weight)
    n = len(sub)
    top = y + h * 0.52
    bot = y + h * 0.14
    if n == 1:
        ys = [(top + bot) / 2]
    else:
        step = (top - bot) / (n - 1)
        ys = [top - step * i for i in range(n)]
    for line, ty in zip(sub, ys):
        _label(ax, x + w / 2, ty, line,
               fs=sub_fs, style="italic",
               color=sub_color or PALETTE["muted"])


def _arrow(ax, p0, p1, *, color=None, lw=1.6, mutation=14,
           connectionstyle="arc3,rad=0", zorder=3):
    color = color or PALETTE["ink"]
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=mutation,
        color=color, lw=lw, connectionstyle=connectionstyle,
        shrinkA=2, shrinkB=2, zorder=zorder,
    ))


# ── Main composition ──────────────────────────────────────────────────
def draw(ax) -> None:
    ax.set_xlim(0, 17.8)
    ax.set_ylim(0, 10.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(PALETTE["bg"])

    # Title block
    _label(ax, 8.90, 9.55,
           "Self-supervised pretraining pipeline",
           fs=16, weight="bold")
    _label(ax, 8.90, 9.10,
           r"SimMIM masked reconstruction  $+$  VICReg invariance  $+$  Fourier auxiliary",
           fs=10.5, style="italic", color=PALETTE["muted"])

    # ── Microscopy patch (left-most, with thumbnail) ─────────────────
    px, py, pw, ph = 0.20, 5.10, 2.10, 2.40
    _rounded(ax, px, py, pw, ph,
             face=PALETTE["input"], edge=PALETTE["input_e"])
    _label(ax, px + pw / 2, py + ph * 0.88,
           "Microscopy patch", fs=10.5, weight="bold")
    _label(ax, px + pw / 2, py + ph * 0.74,
           r"$3 \times 128 \times 128$",
           fs=9.0, style="italic", color=PALETTE["muted"])
    # synthetic thumbnail inside the box
    thumb = _synthetic_microscopy_patch()
    th = ph * 0.45
    tw = th  # square
    tx = px + (pw - tw) / 2
    ty = py + ph * 0.18
    ax.imshow(thumb,
              extent=[tx, tx + tw, ty, ty + th],
              interpolation="bilinear",
              zorder=4, aspect="auto")
    _rounded(ax, tx, ty, tw, th,
             face="none", edge=PALETTE["input_e"], lw=1.0, radius=0.04, zorder=5)
    _label(ax, px + pw / 2, py + ph * 0.07,
           "pre / post / structural",
           fs=8.4, style="italic", color=PALETTE["muted"])

    # ── Two-view augmentation ────────────────────────────────────────
    _box(ax, 2.55, 5.00, 2.5, 2.8,
         title="Two-view augment",
         sub=[
             "flip · rot90 · translate",
             "blur · Poisson + Gaussian",
             "per-channel z-score",
         ],
         face=PALETTE["aug"], edge=PALETTE["aug_e"],
         title_fs=10.5)

    _arrow(ax, (2.30, 6.30), (2.55, 6.30))

    # ── Two augmented views (single stacked box) ─────────────────────
    # Both views feed BOTH branches: each is encoded once clean (VICReg
    # pass) and once after masking (SimMIM pass).
    _box(ax, 5.40, 5.40, 1.7, 2.0,
         title=r"views  $v_1,\, v_2$",
         sub=["two augmented views"],
         face=PALETTE["view"], edge=PALETTE["view_e"],
         title_fs=10.5)
    _arrow(ax, (5.05, 6.30), (5.40, 6.30))

    # branch banners
    _label(ax, 8.40, 8.20,
           "VICReg branch  ·  clean views",
           fs=9.5, weight="bold", color=PALETTE["proj_e"])
    _label(ax, 8.40, 3.85,
           "SimMIM branch  ·  masked views",
           fs=9.5, weight="bold", color=PALETTE["mask_e"])

    # ── Block mask + token (operates on BOTH v1 and v2) ──────────────
    _box(ax, 7.45, 4.45, 2.4, 1.6,
         title="Block mask + token",
         sub=[
             r"ratio $0.6$  ·  $16 {\times} 16$",
             "learnable mask token",
         ],
         face=PALETTE["mask"], edge=PALETTE["mask_e"],
         title_fs=10.5)

    # views -> mask  (masked path, purple)
    _arrow(ax, (7.10, 6.00), (7.45, 5.25),
           color=PALETTE["mask_e"], lw=1.8,
           connectionstyle="arc3,rad=0.20")

    # ── Shared Swin encoder ──────────────────────────────────────────
    enc_x, enc_y, enc_w, enc_h = 10.30, 4.45, 3.30, 3.00
    _stack(ax, enc_x, enc_y, enc_w, enc_h,
           face=PALETTE["encoder"], edge=PALETTE["encoder_e"], layers=3)
    _label(ax, enc_x + enc_w / 2, enc_y + enc_h * 0.88,
           "Swin Transformer", fs=11.5, weight="bold")
    _label(ax, enc_x + enc_w / 2, enc_y + enc_h * 0.74,
           "MONAI SwinTransformer  ·  shared weights",
           fs=8.4, style="italic", color=PALETTE["muted"])
    # static Swin-T spec lines (these do not change across runs)
    _label(ax, enc_x + enc_w / 2, enc_y + enc_h * 0.55,
           r"depths   $(2,\,2,\,6,\,2)$",
           fs=9.0, color=PALETTE["ink"])
    _label(ax, enc_x + enc_w / 2, enc_y + enc_h * 0.43,
           r"heads     $(3,\,6,\,12,\,24)$",
           fs=9.0, color=PALETTE["ink"])
    _label(ax, enc_x + enc_w / 2, enc_y + enc_h * 0.31,
           r"embed   $96 \rightarrow 192 \rightarrow 384 \rightarrow 768$",
           fs=9.0, color=PALETTE["ink"])
    _label(ax, enc_x + enc_w / 2, enc_y + enc_h * 0.13,
           "init: scratch  /  MoBY  /  ImageNet-22k",
           fs=8.6, style="italic", color=PALETTE["encoder_e"])

    # clean views -> encoder top  (VICReg path, blue)
    _arrow(ax, (7.10, 6.80), (enc_x + 0.30, enc_y + enc_h - 0.25),
           color=PALETTE["proj_e"], lw=1.8,
           connectionstyle="arc3,rad=-0.22")
    # block-mask output -> encoder bottom  (SimMIM path, purple)
    _arrow(ax, (9.85, 5.25), (enc_x + 0.30, enc_y + 0.30),
           color=PALETTE["mask_e"], lw=1.8,
           connectionstyle="arc3,rad=0.10")

    # ── Feature taps (z_clean / z_mask) ──────────────────────────────
    feat_x = 13.85
    _box(ax, feat_x, 6.55, 1.10, 0.75,
         title=r"$z_{\mathrm{clean}}$",
         face=PALETTE["feat"], edge=PALETTE["proj_e"],
         title_fs=11)
    _box(ax, feat_x, 4.85, 1.10, 0.75,
         title=r"$z_{\mathrm{mask}}$",
         face=PALETTE["feat"], edge=PALETTE["mask_e"],
         title_fs=11)

    _arrow(ax, (enc_x + enc_w, enc_y + enc_h - 0.55), (feat_x, 6.92),
           color=PALETTE["proj_e"], lw=1.8)
    _arrow(ax, (enc_x + enc_w, enc_y + 0.55), (feat_x, 5.22),
           color=PALETTE["mask_e"], lw=1.8)

    # ── Projector (top branch) — GAP shown as arrow label ──────────
    proj_x = 15.55
    _box(ax, proj_x, 6.20, 2.05, 1.45,
         title="Projector",
         sub=[
             "BN + ReLU MLP",
             "VICReg head",
         ],
         face=PALETTE["proj"], edge=PALETTE["proj_e"],
         title_fs=10.5)
    _arrow(ax, (feat_x + 1.10, 6.92), (proj_x, 6.92),
           color=PALETTE["proj_e"], lw=1.6)
    _label(ax, (feat_x + 1.10 + proj_x) / 2, 7.18,
           "GAP", fs=9.0, weight="bold", color=PALETTE["proj_e"])

    # ── SimMIM decoder (bottom branch) ───────────────────────────────
    _box(ax, proj_x, 4.10, 2.05, 1.85,
         title="SimMIM decoder",
         sub=[
             r"$1{\times}1$  Conv  $\rightarrow$  PixelShuffle",
             "reconstruct masked pixels",
         ],
         face=PALETTE["decoder"], edge=PALETTE["decoder_e"],
         title_fs=10.5)
    _arrow(ax, (feat_x + 1.10, 5.22), (proj_x, 5.22),
           color=PALETTE["mask_e"], lw=1.6)

    # ── Section divider ──────────────────────────────────────────────
    ax.add_patch(FancyArrowPatch(
        (0.5, 3.55), (17.30, 3.55),
        arrowstyle="-", color=PALETTE["rule"], lw=0.9, zorder=0,
    ))
    _label(ax, 8.90, 3.20, "Pretraining objective",
           fs=13, weight="bold")

    # ── Loss boxes ───────────────────────────────────────────────────
    loss_y, loss_h = 1.10, 1.75
    # VICReg
    _box(ax, 0.50, loss_y, 5.50, loss_h,
         title=r"$\mathcal{L}_{\mathrm{VICReg}}$",
         sub=[
             r"$\lambda_{\mathrm{sim}}\,L_{\mathrm{sim}} \; + \; "
             r"\lambda_{\mathrm{std}}\,L_{\mathrm{std}} \; + \; "
             r"\lambda_{\mathrm{cov}}\,L_{\mathrm{cov}}$",
             "invariance + variance + covariance  ·  from projector",
         ],
         face=PALETTE["loss_vic"], edge=PALETTE["loss_vic_e"],
         title_fs=14, sub_fs=9.2, sub_color=PALETTE["ink"])

    # Recon
    _box(ax, 6.20, loss_y, 5.40, loss_h,
         title=r"$\mathcal{L}_{\mathrm{recon}}$",
         sub=[
             "masked L1  ·  foreground-weighted",
             "reconstruct masked pixels (SimMIM)  ·  from decoder",
         ],
         face=PALETTE["loss_recon"], edge=PALETTE["loss_recon_e"],
         title_fs=14, sub_fs=9.2, sub_color=PALETTE["ink"])

    # Fourier
    _box(ax, 11.80, loss_y, 5.50, loss_h,
         title=r"$\mathcal{L}_{\mathrm{fourier}}$",
         sub=[
             "per-tile FFT L1  ·  inspired by CA-MAE",
             "penalises spatial over-smoothing  ·  from decoder",
         ],
         face=PALETTE["loss_fourier"], edge=PALETTE["loss_fourier_e"],
         title_fs=14, sub_fs=9.2, sub_color=PALETTE["ink"])

    # ── Total loss formula ───────────────────────────────────────────
    _label(
        ax, 8.90, 0.40,
        r"$\mathcal{L} \;=\; w_{\mathrm{recon}}\,\mathcal{L}_{\mathrm{recon}} \;+\; "
        r"w_{\mathrm{vicreg}}\,\mathcal{L}_{\mathrm{VICReg}} \;+\; "
        r"w_{\mathrm{fourier}}\,\mathcal{L}_{\mathrm{fourier}}$",
        fs=13)


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
_FIGURES = {
    "pipeline": {
        "fn":     "draw",
        "size":   (14.5, 8.25),
        "fname":  "pretrain_sim_vicreg.svg",
    },
}


def _render(fig_kind: str, out_path: Path) -> None:
    spec = _FIGURES[fig_kind]
    draw_fn = globals()[spec["fn"]]
    w, h = spec["size"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    draw_fn(ax)
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure", choices=("pipeline", "both"),
        default="both",
        help="Which figure to emit.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/imaging_outputs/architecture"),
        help="Output directory (used when --figure=both or when --out omitted).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Explicit output SVG path (only used when --figure != both).",
    )
    args = parser.parse_args()

    if args.figure == "both":
        for kind, spec in _FIGURES.items():
            _render(kind, args.out_dir / spec["fname"])
    else:
        out = args.out or (args.out_dir / _FIGURES[args.figure]["fname"])
        _render(args.figure, out)


if __name__ == "__main__":
    main()
