import numpy as np
from skimage.metrics import structural_similarity

from microssim.ssim.ssim_utils import (
    _scaled_ssim,
    _ssim_with_c3,
    compute_scaled_ssim,
    compute_ssim_elements,
)


def test_compute_ssim(data_range, ground_truth, prediction):
    """Test that MicroSSIM SSIM agrees with the skimage SSIM."""
    # Compute the SSIM using MicroSSIM version
    ssim_elements = compute_ssim_elements(
        ground_truth, prediction, data_range=data_range
    )
    ssim = compute_scaled_ssim(ssim_elements)

    # Compute the SSIM using skimage
    ssim_skimage = structural_similarity(
        ground_truth, prediction, data_range=data_range
    )

    # Check that the two SSIM values are equal
    assert np.isclose(ssim, ssim_skimage, atol=1e-4)


def test_ssim_usual_C3(data_range, ground_truth, prediction):
    """Test that c3=c2/2 leads to the same result."""
    # Compute the SSIM elements
    ssim_elements = compute_ssim_elements(
        ground_truth, prediction, data_range=data_range
    )

    # Compute SSIM with C3=C2/2 and without C3
    ssim_without_c3 = _scaled_ssim(alpha=1.0, elements=ssim_elements)
    ssim_with_c3 = _ssim_with_c3(
        alpha=1.0, elements=ssim_elements, C3=ssim_elements.C2 / 2
    )

    assert np.allclose(ssim_without_c3.SSIM, ssim_with_c3.SSIM)


def test_ssim_c3(data_range, ground_truth, prediction):
    """Test that c3!=c2/2 leads to different results."""
    # Compute the SSIM elements
    ssim_elements = compute_ssim_elements(
        ground_truth, prediction, data_range=data_range
    )

    # Compute SSIM with C3=C2/2 and without C3
    ssim_without_c3 = _scaled_ssim(alpha=1.0, elements=ssim_elements)
    ssim_with_c3 = _ssim_with_c3(
        alpha=1.0, elements=ssim_elements, C3=ssim_elements.C2 * 2
    )

    assert not np.allclose(ssim_without_c3.SSIM, ssim_with_c3.SSIM)


def test_alpha_changes_result(data_range, ground_truth, prediction):
    """Test that alpha!=1. changes the result."""
    # Compute the SSIM using MicroSSIM
    ssim_elements = compute_ssim_elements(
        ground_truth, prediction, data_range=data_range
    )

    ssim = compute_scaled_ssim(alpha=1.0, elements=ssim_elements)
    ssim_alpha = compute_scaled_ssim(alpha=0.5, elements=ssim_elements)

    # Check that the two SSIM values are not equal
    assert not np.isclose(ssim, ssim_alpha)
