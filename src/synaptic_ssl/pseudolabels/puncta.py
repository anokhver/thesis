"""Back-compat shim: the LoG detector moved to `puncta_log`.

Existing `from synaptic_ssl.pseudolabels.puncta import ...` imports keep
working; new code should target `puncta_log` (or `puncta_spotiflow`)
explicitly. Spotiflow lives in `puncta_spotiflow`; this shim does NOT
re-export it -- import it from there directly.
"""
from .puncta_log import (  # noqa: F401
    PunctaCfg,
    DEFAULT_PUNCTA_CFG_PRE,
    DEFAULT_PUNCTA_CFG_POST,
    DEFAULT_NEAR_DILATE_PX,
    resolve_cfg,
    detect_puncta_log,
    score_puncta_zscore,
    filter_by_size_shape,
    derive_zscore_floors,
    detect_puncta_channel,
)
from .puncta_common import puncta_to_mask, restrict_puncta_to_near  # noqa: F401
