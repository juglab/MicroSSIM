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

def test_compute_ssim(image_1, image_2):
    """Test that MicroSSIM SSIM agrees with the skimage SSIM."""
    # Compute the SSIM using MicroSSIM version
    ssim_elements = compute_ssim_elements(image_1, image_2, data_range=65535)
    ssim = compute_ssim(ssim_elements)

    # Compute the SSIM using skimage
    ssim_skimage = structural_similarity(image_1, image_2, data_range=65535)

    # Check that the two SSIM values are equal
    assert np.isclose(ssim, ssim_skimage, atol=1e-4)


def test_ssim_w_and_wo_standard_c3image_1(image_1, image_2):
    """Test that c3=c2/2 leads to the same result."""
    # Compute the SSIM elements
    ssim_elements = compute_ssim_elements(image_1, image_2, data_range=65535)
    assert ssim_elements.C3 is None

    # Compute SSIM with C3=C2/2 and without C3
    ssim_without_c3 = _ssim(alpha=1.0, elements=ssim_elements)

    ssim_elements.C3 = ssim_elements.C2 / 2
    ssim_with_c3 = _ssim_with_c3(alpha=1.0, elements=ssim_elements)

    assert np.allclose(ssim_without_c3.SSIM, ssim_with_c3.SSIM)


def test_ssim_w_and_wo_c3(image_1, image_2):
    """Test that c3!=c2/2 leads to different results."""
    # Compute the SSIM elements
    ssim_elements = compute_ssim_elements(image_1, image_2, data_range=65535)
    assert ssim_elements.C3 is None

    # Compute SSIM with C3=C2/2 and without C3
    ssim_without_c3 = _ssim(alpha=1.0, elements=ssim_elements)

    ssim_elements.C3 = ssim_elements.C2 * 2
    ssim_with_c3 = _ssim_with_c3(alpha=1.0, elements=ssim_elements)

    assert not np.allclose(ssim_without_c3.SSIM, ssim_with_c3.SSIM)


def test_alpha_changes_result(image_1, image_2):
    """Test that alpha!=1. changes the result."""
    # Compute the SSIM using MicroSSIM
    ssim_elements = compute_ssim_elements(image_1, image_2, data_range=65535)

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

