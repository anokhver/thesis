"""SSL training infrastructure: config, data, augmentation, losses, scheduling, and visualisation."""

from .config import BaseCfg, DataCfg, ModelCfg, TrainCfg, SSLCfg, dump_config
from .seeding import seed_everything
from .logging import setup_logger, CSVMetricLogger
from .data import build_dataloaders, compute_channel_stats, TransformedSubset, split_train_val
from .augment import MicroscopyTwoViewTransform, ValSingleViewTransform
from .masking import random_block_mask, apply_mask
from .losses import compute_simmim_vicreg_loss, validation_simmim, simmim_recon_loss, fourier_recon_loss, vicreg_terms
from .lr_schedule import param_groups_layer_decay, make_warmup_cosine
from .checkpoints import save_checkpoint, load_checkpoint, find_latest_checkpoint
from .sanity_batch import (
    SANITY_TRAIN_INDICES, SANITY_VAL_INDICES,
    fixed_two_view_batch, fixed_single_view_batch, overfit_on_batch,
)
from .viz import (
    LOSS_LABELS, loss_label,
    plot_two_views, plot_channel_histograms, plot_recon_panel,
    plot_loss_curves, plot_overfit_curves, plot_embedding_2d,
)

__all__ = [
    # config
    "BaseCfg", "DataCfg", "ModelCfg", "TrainCfg", "SSLCfg", "dump_config",
    # seeding
    "seed_everything",
    # logging
    "setup_logger", "CSVMetricLogger",
    # data
    "build_dataloaders", "compute_channel_stats", "TransformedSubset", "split_train_val",
    # augment
    "MicroscopyTwoViewTransform", "ValSingleViewTransform",
    # masking
    "random_block_mask", "apply_mask",
    # losses
    "compute_simmim_vicreg_loss", "validation_simmim",
    "simmim_recon_loss", "fourier_recon_loss", "vicreg_terms",
    # lr_schedule
    "param_groups_layer_decay", "make_warmup_cosine",
    # checkpoints
    "save_checkpoint", "load_checkpoint", "find_latest_checkpoint",
    # sanity_batch
    "SANITY_TRAIN_INDICES", "SANITY_VAL_INDICES",
    "fixed_two_view_batch", "fixed_single_view_batch", "overfit_on_batch",
    # viz
    "LOSS_LABELS", "loss_label",
    "plot_two_views", "plot_channel_histograms", "plot_recon_panel",
    "plot_loss_curves", "plot_overfit_curves", "plot_embedding_2d",
]