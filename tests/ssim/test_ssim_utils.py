import numpy as np
from skimage.metrics import structural_similarity

from microssim.ssim.ssim_utils import (
    compute_ssim_elements,
    compute_ssim
)

def test_compute_ssim():
    """Test that MicroSSIM SSIM agrees with the skimage SSIM."""
    # Create two random images
    rng = np.random.default_rng(42)
    img1 = rng.integers(0, 256, (100, 100))
    img2 = rng.integers(0, 256, (100, 100))

    # Compute the SSIM using MicroSSIM
    ssim_elements = compute_ssim_elements(img1, img2)
    ssim = compute_ssim(ssim_elements)

    # Compute the SSIM using skimage
    ssim_skimage = structural_similarity(img1, img2)

    # Check that the two SSIM values are equal
    assert np.isclose(ssim, ssim_skimage)