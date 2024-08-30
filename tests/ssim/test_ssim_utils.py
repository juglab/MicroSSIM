import numpy as np
from skimage.metrics import structural_similarity

from microssim.ssim.ssim_utils import (
    compute_ssim_elements,
    compute_ssim
)
from microssim.ssim.ssim_utils import (
    _ssim,
    _ssim_with_c3
)

def test_compute_ssim():
    """Test that MicroSSIM SSIM agrees with the skimage SSIM."""
    # Create two random images
    rng = np.random.default_rng(42)
    img1 = rng.integers(0, 255, (100, 100))
    img2 = rng.poisson(img1)

    # Compute the SSIM using MicroSSIM version
    ssim_elements = compute_ssim_elements(img1, img2, data_range=255)
    ssim = compute_ssim(ssim_elements)

    # Compute the SSIM using skimage
    ssim_skimage = structural_similarity(img1, img2, data_range=255)

    # Check that the two SSIM values are equal
    assert np.isclose(ssim, ssim_skimage)


def test_ssim_w_and_wo_standard_c3():
    """Test that c3=c2/2 leads to the same result."""
    # Create two random images
    rng = np.random.default_rng(42)
    img1 = rng.integers(0, 255, (100, 100))
    img2 = rng.poisson(img1)

    # Compute the SSIM elements
    ssim_elements = compute_ssim_elements(img1, img2, data_range=255)
    assert ssim_elements.C3 is None

    # Compute SSIM with C3=C2/2 and without C3
    ssim_without_c3 = _ssim(alpha=1.0, elements=ssim_elements)

    ssim_elements.C3 = ssim_elements.C2 / 2
    ssim_with_c3 = _ssim_with_c3(alpha=1.0, elements=ssim_elements)

    assert np.allclose(ssim_without_c3.SSIM, ssim_with_c3.SSIM)


def test_ssim_w_and_wo_c3():
    """Test that c3!=c2/2 leads to different results."""
    # Create two random images
    rng = np.random.default_rng(42)
    img1 = rng.integers(0, 255, (100, 100))
    img2 = rng.poisson(img1)

    # Compute the SSIM elements
    ssim_elements = compute_ssim_elements(img1, img2, data_range=255)
    assert ssim_elements.C3 is None

    # Compute SSIM with C3=C2/2 and without C3
    ssim_without_c3 = _ssim(alpha=1.0, elements=ssim_elements)

    ssim_elements.C3 = ssim_elements.C2 * 2
    ssim_with_c3 = _ssim_with_c3(alpha=1.0, elements=ssim_elements)

    assert not np.allclose(ssim_without_c3.SSIM, ssim_with_c3.SSIM)


def test_alpha_changes_result():
    """Test that alpha!=1. changes the result."""
    # Create two random images
    rng = np.random.default_rng(42)
    img1 = rng.integers(0, 255, (100, 100))
    img2 = rng.poisson(img1)

    # Compute the SSIM using MicroSSIM
    ssim_elements = compute_ssim_elements(img1, img2, data_range=255)

    ssim = compute_ssim(alpha=1.0, elements=ssim_elements)
    ssim_alpha = compute_ssim(alpha=.5, elements=ssim_elements)

    # Check that the two SSIM values are not equal
    assert not np.isclose(ssim, ssim_alpha)

    # Same when specifying C3
    ssim_elements.C3 = ssim_elements.C2 * 2
    
    ssim = compute_ssim(alpha=1.0, elements=ssim_elements)
    ssim_alpha = compute_ssim(alpha=.5, elements=ssim_elements)

    # Check that the two SSIM values are not equal
    assert not np.isclose(ssim, ssim_alpha)

