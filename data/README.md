# `data/`

Datasets, intermediate artefacts, and training outputs. `data/` is
git-ignored; only the placeholders and small reference files listed
below are tracked. Everything else lives on local disk or the cluster.

## Layout

```
data/
├── .gitkeep
├── README.md                          # this file
│
├── Microscopy/                        # raw acquisitions (.vsi/.ets/.tif/.oex)
├── Microscopy_meta/                   # batch_extract_metadata.py output
│
├── patches_128/                       # 128x128 patches + index.csv (SSL training input)
├── patches_32/                        # 32x32 patches  + index.csv
│
├── metadata/Microscopy_meta/          # session metadata (TRACKED)
│   ├── metadata.csv                   #   one row per acquisition session
│   └── metadata_full.json             #   full raw metadata for auditing
│
├── checkpoints/                       # pretrained encoder weights
│   └── README.md                      #   download links (Swin-T 22k, MoBY) — TRACKED
│
├── embeddings/                        # cached encoder embeddings (.npy) for clustering
├── imaging_outputs/                   # figures and renders from `imaging/` scripts
│
├── training_outputs/                  # one timestamped subdir per training run
│   ├── pretrain_moby/.gitkeep         #   placeholders — TRACKED
│   ├── pretrain_scratch/.gitkeep
│   ├── pretrain_tinny22k/.gitkeep
│   ├── segmentation/.gitkeep          #   SwinUNETR joint 2-channel runs
│   ├── encoder_audit/.gitkeep         #   encoder_feature_audit.py outputs
│   ├── clustering/.gitkeep
│   └── segmentation/.gitkeep
│
└── csv/                               # external CSV exports (manual)
```

## What gets written here

| Producer                                            | Output |
|:----------------------------------------------------|:-------|
| `scripts/metadata/batch_extract_metadata.py`        | `Microscopy_meta/metadata.csv` + `metadata_full.json`. |
| `scripts/preprocess/batch_preprocess.py`            | `patches_128/<session>/*.npy` + per-folder `index.csv`. |
| `scripts/preprocess/tile_from_mip.py`               | `patches_128_from_zip/<date>/...` from pre-MIPped `.npy`. |
| `scripts/training/pretrain_simmim_vicreg.py`        | `training_outputs/<run_label>/` (config, checkpoints, metrics, plots). |
| `scripts/training/train_swinunetr_joint_2ch.py`     | `training_outputs/segmentation/<run_label>/` (SwinUNETR checkpoints + plots). |
| `scripts/evaluation/extract_embeddings.py`          | `embeddings/<run>.npy` (or `training_outputs/<run>/embeddings.npy`). |
| `scripts/run_clustering.py`                         | `training_outputs/clustering/<run>/` (Leiden + PERMANOVA + plots). |
| `scripts/pseudolabels/*_from_mip.py`                | Pseudo-label masks alongside the source MIPs. |
| `imaging/preview_*.py`, `imaging/draw_*.py`        | `imaging_outputs/...` figures. |

## Pretrained weights

`checkpoints/README.md` has the upstream URLs (Swin-Tiny ImageNet-22k,
MoBY contrastive). Drop the `.pth` / `.pt` files into `data/checkpoints/`
and point `base.pretrained_ckpt_path` at them.
