import numpy as np

from microssim.image_processing.micro_ssim_normalization import (
    compute_norm_parameters,
    normalize_min_max,
)


def test_compute_passing_all_params(ground_truth, prediction):
    """Test that the function returns the same parameters if they are all provided."""
    offset_gt = 10.0
    offset_pred = 5.0
    max_val = 100.0
    assert compute_norm_parameters(
        ground_truth,
        prediction,
        offset_gt=offset_gt,
        offset_pred=offset_pred,
        max_val=max_val,
    ) == (offset_gt, offset_pred, max_val)


def test_compute_params(ground_truth, prediction):
    """Test that the function returns the correct parameters."""
    bg_percentile = 5
    offset_gt = np.percentile(ground_truth, bg_percentile, keepdims=False)
    offset_pred = np.percentile(prediction, bg_percentile, keepdims=False)
    max_val = np.max(ground_truth - offset_gt)

    assert compute_norm_parameters(
        ground_truth,
        prediction,
        bg_percentile=bg_percentile,
    ) == (offset_gt, offset_pred, max_val)


def test_normalize_min_max():
    """Test that the function normalizes an array correctly."""
    min_val = 20
    max_val = 100
    ground_truth = min_val + np.arange(100).reshape(10, 10)

    normalized = normalize_min_max(ground_truth, min_val, max_val)

    assert np.isclose(normalized.min(), 0)
    assert np.isclose(normalized.max(), 1, atol=1e-2)


def test_normalize_min_max_list():
    """Test that the function normalizes an array correctly."""
    lst = []

    for i in range(3, 6):
        min_val = i * 20
        max_val = (i * 10) ** 2
        image = min_val + np.arange((i * 10) ** 2).reshape(i * 10, i * 10)

        normalized = normalize_min_max(image, min_val, max_val)

        lst.append(normalized)

    assert np.allclose(normalized.min(), 0)
    assert np.allclose(normalized.max(), 1, atol=1e-3)
