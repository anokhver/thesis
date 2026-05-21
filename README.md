# Self-supervised segmentation of synaptic structures in fluorescence microscopy

Bachelor's thesis code. SSL pretraining of a Swin-Tiny encoder on
3-channel confocal microscopy, then downstream segmentation of synaptic
puncta.

## Overview

```
Raw microscopy  ->  Patch extraction  ->  SSL pretraining  ->  Segmentation
 (.czi/.ets/.vsi)    (128x128 / 32x32)    (SimMIM + VICReg)     (SwinUNETR)
```

Encoder: MONAI Swin-Tiny. Pretraining: SimMIM (masked image modeling)
plus VICReg (variance-invariance-covariance), with an optional Fourier
auxiliary loss. Init can be random, MoBY contrastive, or Swin-T
ImageNet-22k. The pretrained encoder is then fine-tuned with SwinUNETR
for pixel-level segmentation of pre- and post-synaptic puncta.

## Repository structure

```
thesis/
├── pyproject.toml                # Python package metadata (pip install -e .)
├── requirements.txt              # Pinned dev dependencies
├── environment.yml               # Minimal conda env (delegates to requirements.txt)
├── environment.windows.yml       # Same, for Windows hosts
├── README.md
│
├── configs/                      # JSON configuration files (see "Configuration" below)
│   ├── preprocess/               #   Patch extraction (128 / 32)
│   ├── pretrain_moby/            #   Pretrain init: MoBY contrastive Swin-T
│   ├── pretrain_scratch/         #   Pretrain init: random
│   └── pretrain_tinny/           #   Pretrain init: Swin-T ImageNet-22k (typo: "tinny" → "tiny22k")
│
├── notebooks/                    # Jupyter notebooks
│   ├── loaded-weights/           #   MoBY- and tiny22k-initialised pretraining
│   ├── from-scratch/             #   From-scratch pretraining
│   ├── clustering/               #   Embedding clustering pipeline
│   ├── segmentation/             #   Downstream SwinUNETR fine-tuning
│   └── pseudolabels/             #   Blob pseudo-label generation
│       └── dendrite_methods/     #   Dendrite-mask experiments (FDT, KDE, MST, ...)
│
├── tests/                        # Manual validation notebooks
│
├── scripts/                      # Headless entry points + cluster jobs (see "Scripts" below)
│   ├── preprocess/               #   Patch tiling, preprocessing orchestration, noise scoring
│   ├── metadata/                 #   Microscope/experiment metadata extraction
│   ├── training/                 #   SSL pretraining + post-training recovery
│   ├── evaluation/               #   Embedding extraction from a trained checkpoint
│   ├── metacentrum/              #   PBS submission scripts (Metacentrum / CESNET)
│   └── nudz/                     #   SLURM scripts + Windows .bat/.ps1 wrappers (NUDZ cluster)
│
├── imaging/                      # Microscopy browsing + figure helpers (NOT in pip package)
│   └── README.md                 #   imaging/ contents — see there
│
├── data/                         # Raw + processed data (NOT in package, NOT in git)
│   └── README.md                 #   expected layout — see there
│
└── src/synaptic_ssl/             # Python package source (`pip install -e .`)
    │
    ├── models/                   # Model definitions
    │   ├── swin.py               #   Swin encoder + SimMIM/VICReg heads
    │   └── weight_loading.py     #   Pretrained weight transfer (MoBY, timm)
    │
    ├── training/                 # Reusable training utilities
    │   ├── config.py             #   Dataclass configs (BaseCfg, DataCfg, ...)
    │   ├── losses.py             #   SimMIM + VICReg + Fourier loss
    │   ├── augment.py            #   Microscopy-specific augmentations
    │   ├── masking.py            #   Block masking for SimMIM
    │   ├── lr_schedule.py        #   Warmup-cosine LR + layer-wise decay
    │   ├── checkpoints.py        #   Save/load/resume checkpoints
    │   ├── sanity_batch.py       #   Fixed-batch overfit checks
    │   ├── viz.py                #   Training visualizations
    │   ├── data.py               #   DataLoader construction
    │   ├── logging.py            #   File + CSV metric logging
    │   └── seeding.py            #   Reproducibility
    │
    ├── ssl_training/             # SSL pretraining workflow
    │   ├── post_training.py      #   Post-training reconstruction & curves
    │   ├── train_loop.py         #   Per-epoch train/validate
    │   ├── unfreeze.py           #   Gradual unfreezing schedule
    │   ├── full_image_recon.py   #   Sliding-window full-image reconstruction
    │   └── config_loading.py     #   JSON config loading + validation
    │
    ├── utils_data/               # Data processing
    │   ├── preprocess_training.py #  Raw → MIP → normalize → tile patches
    │   ├── noise_detection.py    #   Flag noise-dominated source images (frequency-domain)
    │   ├── patch_dataset.py      #   PyTorch Dataset for .npy patches
    │   ├── reassemble.py         #   ImageCache + reassemble + load_patch_records
    │   ├── load_data.py          #   Microscopy file loading utilities
    │   └── split.py              #   Train/val splitting
    │
    ├── segmentation/             # Downstream segmentation
    │   ├── model.py              #   SwinUNETR wrapper + encoder weight loading
    │   ├── dataset.py            #   Pseudo-label segmentation dataset
    │   ├── losses.py             #   Dice / BCE losses + Dice metric
    │   ├── augment.py            #   Segmentation augmentations
    │   ├── inference.py          #   Sliding-window prediction
    │   └── config.py             #   Segmentation training config
    │
    ├── clustering/               # Embedding analysis
    │   ├── core.py               #   Pooled embedding extraction + PCA/UMAP
    │   ├── pipeline.py           #   HPL-style clustering pipeline (Leiden + PERMANOVA)
    │   └── viz.py                #   Cluster visualizations
    │
    └── pseudolabels/             # Pseudo-label generation
        ├── blobs.py              #   LoG blob detection + dendrite/soma masks
        ├── refine.py             #   Iterative region growing + fusion
        └── viz.py                #   Pseudo-label visualization
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .                    # makes `training`, `models`, etc. importable
```

### 2. Preprocess raw microscopy into patches

```bash
# 128×128 patches
python src/synaptic_ssl/utils_data/preprocess_training.py --config configs/preprocess/patches_128.json

# 32×32 patches
python src/synaptic_ssl/utils_data/preprocess_training.py --config configs/preprocess/patches_32.json
```

### 3. Run SSL pretraining

**Script mode** (recommended for cluster jobs):

```bash
# MoBY-initialized, 128×128 (default config)
python scripts/training/pretrain_simmim_vicreg.py --config configs/pretrain_moby/128_default.json

# From scratch, 128×128, VICReg off
python scripts/training/pretrain_simmim_vicreg.py --config configs/pretrain_scratch/128_no_fourier_vicreg_off.json

# Dry run (validate config without training)
python scripts/training/pretrain_simmim_vicreg.py --config configs/pretrain_moby/128_default.json --dry-run

# Resume interrupted run
python scripts/training/pretrain_simmim_vicreg.py --config configs/pretrain_moby/128_default.json --resume data/training_outputs/my_run/last.pt
```

**Notebook mode** (interactive, with inline plots):
- `notebooks/loaded-weights/pretrain_simmim_vicreg_moby.ipynb`
- `notebooks/loaded-weights/pretrain_simmim_vicreg_tiny22k.ipynb`
- `notebooks/from-scratch/pretrain_simmim_vicreg.ipynb`

### 4. Cluster submission

PBS (Metacentrum / CESNET):
```bash
qsub scripts/metacentrum/submit_pretrain_moby_py.sh                  # script-based, default config
qsub -v CONFIG=configs/pretrain_moby/128_no_fourier_vicreg_on.json \
     scripts/metacentrum/submit_pretrain_moby_py.sh                  # override config
qsub scripts/metacentrum/submit_loaded_moby.sh                       # notebook-based (papermill)
qsub scripts/metacentrum/submit_from_scratch.sh
qsub scripts/metacentrum/submit_preprocess.sh
qsub scripts/metacentrum/submit_all.sh [all|loadedmoby|scratch|cluster|chain]
```

SLURM (NUDZ):
```bash
sbatch scripts/nudz/submit_pretrain_moby_py.slurm.sh
sbatch scripts/nudz/submit_from_scratch.slurm.sh
bash   scripts/nudz/submit_all.sh   # convenience wrapper, calls sbatch
```

Windows (NUDZ workstation, pre-staging from a mounted disk):
```powershell
scripts\nudz\run_extract_metadata_windows.ps1 -Mode smoke
scripts\nudz\run_preprocess_windows.ps1       -Mode full -Workers 4
scripts\nudz\run_preprocess_notile_windows.ps1            # full-MIP, no tiling
```

## Web frontend (Docker)

The Django UI (`Synaptic Cluster Explorer`) runs the embedding-extract +
Leiden-clustering half of the pipeline through a browser. The whole stack
ships as a single image.

```bash
# Production-style: gunicorn + WhiteNoise inside the image.
cp .env.example .env                          # edit secrets, point paths
docker compose -f docker-compose.yml up --build

# Dev with hot-reload (auto-loads docker-compose.override.yml):
docker compose up --build
```

Browse to <http://localhost:8000/>. Liveness probe at `/healthz/`.

The image **does not bake in any dataset.** Every storage path is
overridable so the same image can talk to local volumes or a lab share:

| Env var              | Default                | Purpose                                  |
|----------------------|------------------------|------------------------------------------|
| `SCE_MEDIA_ROOT`     | `/app/media`           | Root of run artefacts, uploads, plots.   |
| `SCE_CHECKPOINT_DIR` | `$SCE_MEDIA_ROOT/checkpoints` | Encoder `.pt` files.                  |
| `SCE_RUNS_DIR`       | `$SCE_MEDIA_ROOT/runs` | Per-run input/bundle/output trees.       |
| `SCE_SQLITE_PATH`    | `/app/db/db.sqlite3`   | App database.                            |

The compose file mounts the in-image `/app/db` and `/app/media` to named
Docker volumes so SQLite + uploads survive container restarts. To mount
an external disk instead, set the env vars above to paths inside `/data`
(or wherever) and add a bind volume for that path:

```yaml
services:
  web:
    environment:
      SCE_MEDIA_ROOT: /data/media
      SCE_CHECKPOINT_DIR: /data/checkpoints
      SCE_RUNS_DIR: /data/runs
    volumes:
      - /mnt/lab_share/synapseg:/data
```

GPU note: the image is CPU-only by default. PyTorch falls back to CPU
when no NVIDIA driver is visible (`torch.cuda.is_available() == False`).
To run on GPU, install `nvidia-container-toolkit` on the host and add
`runtime: nvidia` (Compose v2 `gpus: all`) to the `web` service.

## Scripts

### Top-level Python entry points (`scripts/<category>/*.py`)

| Script                                       | Purpose |
|:---------------------------------------------|:--------|
| `preprocess/batch_preprocess.py`             | Drive `preprocess_training.py` over every session folder under a root. |
| `preprocess/tile_from_mip.py`                | Tile pre-saved full-MIP `.npy` files into patches (folder, parent-of-folders, or `.zip` archive — pick via `--input_dir` / `--input_root` / `--zip`). |
| `preprocess/score_image_noise.py`            | Flag noise-dominated source images and emit a denylist for `PatchDataset`. |
| `metadata/batch_extract_metadata.py`         | One-pass extraction of microscope metadata per session (`metadata.csv`, `metadata_full.json`). |
| `metadata/gather_experiment_metadata.py`     | Aggregate per-experiment metadata (parsed names + patch index + optional Olympus enrichment). |
| `metadata/scan_z_range_all_images.py`        | Per-file z-bounds scan over a microscopy tree. |
| `training/pretrain_simmim_vicreg.py`         | Headless SimMIM + VICReg pretraining (equivalent to the notebooks). |
| `training/recover_post_training_viz.py`      | Regenerate post-training plots from a completed/partial run folder. |
| `evaluation/extract_embeddings.py`           | Cache pooled encoder embeddings for downstream clustering. |

### Cluster submission

| Directory                  | Scheduler      | Notes |
|:---------------------------|:---------------|:------|
| `scripts/metacentrum/`     | PBS (`qsub`)   | Metacentrum / CESNET. `submit_all.sh` chains loaded-moby -> clustering. |
| `scripts/nudz/*.slurm.sh`  | SLURM (`sbatch`) | NUDZ cluster. `submit_all.sh` mirrors the PBS one. |
| `scripts/nudz/*.{bat,ps1}` | Windows shell  | Local pre-staging from the lab's Z: drive (metadata extraction, preprocessing with/without tiling). |

## Configuration

All configs live in `configs/` as JSON files:

```
configs/
├── preprocess/
│   ├── patches_128.json          # 128×128 patch extraction
│   └── patches_32.json           # 32×32 patch extraction
│
├── pretrain_moby/                # init from MoBY contrastive Swin-T weights
│   ├── 128_default.json          #   SimMIM + VICReg + Fourier  (recommended)
│   ├── 128_no_fourier_vicreg_on.json   # SimMIM + VICReg, no Fourier
│   └── 128_no_fourier_vicreg_off.json  # SimMIM only
│
├── pretrain_scratch/             # random init
│   ├── 128_default.json
│   ├── 128_no_fourier_vicreg_on.json
│   └── 128_no_fourier_vicreg_off.json
│
└── pretrain_tinny/               # init from Swin-T ImageNet-22k weights ("tinny" is a typo of "tiny22k")
    ├── 128_default.json
    ├── 128_no_fourier_vicreg_on.json
    └── 128_no_fourier_vicreg_off.json
```

Each filename names two axes: `<resolution>_<variant>.json`. The
`vicreg_{on,off}` suffix toggles the VICReg joint-embedding branch
(`ssl.w_vicreg`); `no_fourier` disables the Fourier auxiliary loss
(`ssl.w_fourier = 0`). `128_default` keeps both losses on.

32×32 pretraining configs were retired; only the matching
preprocessing config remains under `configs/preprocess/`.

### Key config differences

| Setting                  | `pretrain_moby` | `pretrain_scratch` | `pretrain_tinny` |
|:-------------------------|:---------------:|:------------------:|:----------------:|
| `base.init_source`       | `moby`          | `scratch`          | `tiny22k`        |
| `base.pretrained_ckpt_path` | MoBY `.pth`  | (none)             | Swin-T 22k `.pth` |
| `train.base_lr`          | `1.0e-4`        | `2.0e-4`           | `1.0e-4`         |
| `train.epochs`           | `200`           | `400`              | `200`            |
| `train.freeze_encoder_epochs` | `5`        | `0`                | `5`              |
| `ssl.w_vicreg` (default) | `0.1`           | `0.02`             | `0.1`            |
| `model.dropout_path_rate`| `0.05`          | `0.0`              | `0.05`           |

Inside each preset, the `_on` / `_off` / `default` variants override
`ssl.w_vicreg` and `ssl.w_fourier`; everything else stays at the
preset's defaults.

## Method

### Pretraining: SimMIM + VICReg

- **SimMIM** (Xie et al., CVPR 2022): block masking + pixel-space
  reconstruction. Random patches of the input are masked; the decoder
  predicts pixel values.

- **VICReg** (Bardes et al., ICLR 2022): variance-invariance-covariance
  regularisation on pooled embeddings from two augmented views. No
  negative pairs needed.

- **Fourier auxiliary loss** (inspired by CA-MAE, Kraus et al., CVPR
  2024): optional FFT-domain L1 on masked tiles, for high-frequency
  texture. CA-MAE compares FFT magnitudes only; this code compares
  full complex spectra (`|F_pred - F_target|`), so within-tile
  translations are also penalised.

### Architecture

- **Encoder**: MONAI `SwinTransformer` (Swin-Tiny: depths `[2,2,6,2]`, heads `[3,6,12,24]`, 96-dim features)
- **Decoder**: 1x1 Conv + PixelShuffle upsampler for SimMIM reconstruction (functionally a linear prediction head, per SimMIM §3.3)
- **Projector**: 3-layer MLP expander for VICReg embeddings
- **Input**: 3-channel fluorescence microscopy (pre-synaptic, post-synaptic, structural)

### Weight initialization

| Source | Description |
|:-------|:------------|
| `scratch` | Random init (PyTorch / MONAI defaults) |
| `moby`    | MoBY contrastive self-supervised Swin-T (Xie et al., 2021) |
| `tiny22k` | Swin-T ImageNet-22k (timm) |

### Training features

- Gradual unfreezing (MoBY init): layers unfrozen top-down over the warmup period.
- Layer-wise LR decay (optional): lower LR for earlier layers.
- AMP mixed precision via `GradScaler`.
- Foreground-weighted reconstruction (optional): each masked pixel's L1
  error is scaled by `1 + alpha * sigmoid((target - tau) / temp)` so
  bright structures dominate the gradient. Off by default
  (`fg_weight_alpha = 0`). VICReg branch uses plain global average
  pooling regardless.
- Checkpointing: best model, last model, periodic snapshots.

## Outputs

Each training run produces a timestamped directory under
`base.output_root` (default `data/training_outputs/`):

```
data/training_outputs/simmim_vicreg_pretrain_moby_128_20260515_191500/
├── config.json                 # Full config snapshot
├── channel_stats.json          # Per-channel mean/std
├── run.log                     # Detailed log
├── metrics.csv                 # Per-epoch metrics
├── best_model.pt               # Best checkpoint (full state)
├── last.pt                     # Latest checkpoint
├── epoch_0025.pt               # Periodic snapshots
├── pretrained_encoder_*.pt     # Encoder-only (for downstream)
├── sanity_two_views.png        # Augmentation visualization
├── sanity_histograms.png       # Channel distributions
├── sanity_curves_overfit.png   # Overfit check curves
├── sanity_recon_overfit.png    # Overfit reconstruction
├── post_recon_train.png        # Post-training reconstruction (train)
├── post_recon_val.png          # Post-training reconstruction (val)
└── post_curves.png             # Loss curves
```

## Evaluation

### Extract embeddings

```bash
python scripts/evaluation/extract_embeddings.py \
    --checkpoint data/training_outputs/my_run/best_model.pt \
    --data-root  data/patches_128 \
    --output     data/training_outputs/my_run/embeddings.npy
```

### Clustering analysis

Notebooks under `notebooks/clustering/` do PCA, UMAP, and Leiden on the
extracted embeddings.

## References

- SimMIM — Xie et al., "SimMIM: A Simple Framework for Masked Image Modeling", CVPR 2022.
- VICReg — Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning", ICLR 2022.
- MoBY — Xie et al., "Self-Supervised Learning with Swin Transformers", arXiv:2105.04553, 2021.
- CA-MAE — Kraus et al., "Masked Autoencoders for Microscopy are Scalable Learners of Cellular Biology", CVPR 2024.
- VasoMIM — Huang et al., "Masked Image Modeling for Vascular Segmentation", AAAI 2026.
- SwinUNETR — Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors", BrainLes 2021.
- AnatoMask — Li et al., "AnatoMask: Enhancing Medical Image Segmentation with Reconstruction-guided Self-pretraining", ECCV 2024.
