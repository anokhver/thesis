"""SSL pretraining helpers: config loading, training loop, post-training viz."""

from .config_loading import load_config
from .full_image_recon import full_image_sliding_recon, run_full_image_recon
from .post_training import (
    build_run_label,
    eval_recon_batch,
    reload_best_checkpoint,
    post_training_reconstruction,
    plot_post_training_curves,
)
from .runner import (
    DataState,
    ModelState,
    OptimState,
    setup_data,
    build_model,
    setup_optimizer,
    run_sanity_checks,
    run_overfit_check,
    training_loop,
    post_training_flow,
)
from .train_loop import (
    all_trainable_params,
    heads_iter_lrs,
    set_train,
    train_one_epoch,
    validate_one_epoch,
)
from .unfreeze import (
    DEFAULT_SCRATCH_UNFREEZE_SCHEDULE,
    DEFAULT_UNFREEZE_SCHEDULE,
    VALID_GROUPS,
    apply_unfreeze_schedule,
    default_unfreeze_schedule,
    make_phase_tag,
)

__all__ = [
    # config loading
    "load_config",
    # unfreeze schedule
    "DEFAULT_SCRATCH_UNFREEZE_SCHEDULE",
    "DEFAULT_UNFREEZE_SCHEDULE",
    "VALID_GROUPS",
    "apply_unfreeze_schedule",
    "default_unfreeze_schedule",
    "make_phase_tag",
    # training loop
    "all_trainable_params",
    "heads_iter_lrs",
    "set_train",
    "train_one_epoch",
    "validate_one_epoch",
    # post-training
    "build_run_label",
    "eval_recon_batch",
    "reload_best_checkpoint",
    "post_training_reconstruction",
    "plot_post_training_curves",
    "full_image_sliding_recon",
    "run_full_image_recon",
    # runner
    "DataState", "ModelState", "OptimState",
    "setup_data", "build_model", "setup_optimizer",
    "run_sanity_checks", "run_overfit_check",
    "training_loop", "post_training_flow",
]