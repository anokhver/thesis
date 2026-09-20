import numpy as np
import pytest

from synaptic_ssl.pseudolabels.puncta_common import mask_overlap
from synaptic_ssl.segmentation.inference import postprocess_joint_probability_map


def test_mask_overlap_counts_and_mask():
    pre = np.zeros((12, 12), dtype=bool)
    post = np.zeros_like(pre)
    pre[2:5, 2:5] = True
    post[2:5, 2:5] = True
    pre[8:10, 8:10] = True

    overlap, n_pre, n_post, n_pairs = mask_overlap(pre, post)

    assert (n_pre, n_post, n_pairs) == (2, 1, 1)
    assert int(overlap.sum()) == 9
    assert np.array_equal(overlap, pre & post)


def test_mask_overlap_uses_fixed_distance_rule():
    pre = np.zeros((10, 10), dtype=bool)
    post = np.zeros_like(pre)
    pre[2:6, 2:6] = True
    post[4:7, 5:8] = True

    overlap, _, _, n_pairs = mask_overlap(pre, post)

    assert n_pairs == 0
    assert not overlap.any()


def test_mask_overlap_matches_close_components_one_to_one():
    pre = np.zeros((12, 12), dtype=bool)
    post = np.zeros_like(pre)
    pre[2:4, 2:4] = True
    pre[7:9, 7:9] = True
    post[3:5, 3:5] = True
    post[8:10, 8:10] = True

    overlap, _, _, n_pairs = mask_overlap(pre, post, max_distance_px=2.36)

    assert n_pairs == 2
    assert np.array_equal(overlap, pre | post)


def test_mask_overlap_does_not_reuse_components():
    pre = np.zeros((12, 12), dtype=bool)
    post = np.zeros_like(pre)
    pre[4:6, 4:6] = True
    post[3:5, 3:5] = True
    post[6:8, 6:8] = True

    _, _, _, n_pairs = mask_overlap(pre, post, max_distance_px=2.36)

    assert n_pairs == 1


def test_mask_overlap_supports_legacy_fraction_rule():
    pre = np.zeros((10, 10), dtype=bool)
    post = np.zeros_like(pre)
    pre[2:6, 2:6] = True
    post[4:7, 4:7] = True

    overlap, _, _, n_pairs = mask_overlap(
        pre,
        post,
        min_fraction=0.8,
        rule="smaller_component_fraction",
    )

    assert n_pairs == 0
    assert not overlap.any()


def test_probability_postprocessing_returns_masks_and_counts():
    probs = np.zeros((2, 10, 10), dtype=np.float32)
    probs[0, 2:5, 2:5] = 0.9
    probs[1, 2:5, 2:5] = 0.9

    result = postprocess_joint_probability_map(probs)

    assert result.pre_mask.sum() == 9
    assert result.post_mask.sum() == 9
    assert result.overlap_mask.sum() == 9
    assert (result.n_pre_components, result.n_post_components) == (1, 1)
    assert result.n_overlap_components == 1


@pytest.mark.parametrize("bad_shape", [(10, 10), (1, 10, 10), (3, 10, 10)])
def test_probability_postprocessing_rejects_wrong_channel_shape(bad_shape):
    with pytest.raises(ValueError):
        postprocess_joint_probability_map(np.zeros(bad_shape, dtype=np.float32))