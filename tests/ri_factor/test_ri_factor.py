import numpy as np
import pytest

from microssim.ri_factor.ri_factor import get_global_ri_factor, get_ri_factor
from microssim.ssim.ssim_utils import (
    compute_ssim_elements,
)


@pytest.mark.parametrize("scaling", [1, 10, 100])
def test_get_ri_factor(data_range, image_1, image_2, scaling):
    """Test the range invariant factor is almost equal to the scaling applied."""
    # Compute the SSIM elements
    ssim_elements = compute_ssim_elements(
        image_1, 10 * scaling * image_2, data_range=data_range
    )

    # Compute the RI factor
    ri_factor = get_ri_factor(ssim_elements)
    assert np.isclose(abs(ri_factor * scaling), 1.0, atol=1e-2)


# TODO: implement tests
@pytest.mark.parametrize("scaling", [1, 10, 100])
def test_global_ri_factor(data_range, image_1, image_2, scaling):
    """Test the range invariant factor is almost equal to the scaling applied."""
    # Linearize arrays
    image_1 = image_1.reshape(-1)
    image_2 = (10 * scaling * image_2).reshape(-1)

    # Compute the global RI factor
    global_ri_factor = get_global_ri_factor(image_1, image_2)
    assert np.isclose(global_ri_factor, 1.0, atol=1e-2)
