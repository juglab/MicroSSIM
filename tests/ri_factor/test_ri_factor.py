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


def test_global_ri_different_types(image_1, image_2):
    """Test that an error is raised if the two images have different types."""
    with pytest.raises(ValueError):
        get_global_ri_factor(image_1, [image_2])


def test_global_ri_different_shapes(image_1, image_2):
    """Test that an error is raised if the two images have different shapes."""
    with pytest.raises(ValueError):
        get_global_ri_factor(image_1, image_2[:10])


def test_global_ri_different_lengths(image_1, image_2):
    """Test that an error is raised if the two images have different lengths."""
    with pytest.raises(ValueError):
        get_global_ri_factor([image_1], [image_2, image_2])


# TODO: implement tests for global ri factor
