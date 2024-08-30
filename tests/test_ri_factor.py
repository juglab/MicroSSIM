import pytest
import numpy as np

from microssim._ri_factor import get_ri_factor
from microssim.ssim.ssim_utils import (
    compute_ssim_elements,
)

@pytest.mark.parametrize("scaling", [1, 10, 100])
def test_compute_ri(scaling):
    """Test the range invariant factor is almost equal to the scaling applied."""
    # Create two random images
    rng = np.random.default_rng(42)
    img1 = rng.integers(0, 255, (100, 100))
    img2 = scaling * rng.poisson(img1)

    # Compute the SSIM elements
    ssim_elements = compute_ssim_elements(img1, img2, data_range=255)

    # Compute the RI factor
    ri_factor = get_ri_factor(ssim_elements)
    assert np.isclose(abs(ri_factor * scaling), 1.0, atol=1e-2)
