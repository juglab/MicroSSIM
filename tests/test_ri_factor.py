import pytest
import numpy as np

from microssim._ri_factor import get_ri_factor
from microssim.ssim.ssim_utils import (
    compute_ssim_elements,
)

@pytest.mark.parametrize("scaling", [1, 10, 100])
def test_compute_ri(image_1, image_2, scaling):
    """Test the range invariant factor is almost equal to the scaling applied."""
    # Compute the SSIM elements
    ssim_elements = compute_ssim_elements(
        image_1, 10 * scaling * image_2, data_range=255
    )

    # Compute the RI factor
    ri_factor = get_ri_factor(ssim_elements)
    assert np.isclose(abs(ri_factor * scaling), 1.0, atol=1e-2)
