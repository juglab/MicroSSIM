import numpy as np
import pytest

from microssim.ri_factor.ri_factor import get_global_ri_factor, get_ri_factor
from microssim.ssim.ssim_utils import (
    compute_ssim_elements,
)


@pytest.mark.parametrize("scaling", [1, 10, 100])
def test_get_ri_factor(data_range, ground_truth, prediction, scaling):
    """Test the range invariant factor is almost equal to the scaling applied."""
    # Compute the SSIM elements
    ssim_elements = compute_ssim_elements(
        ground_truth, 10 * scaling * prediction, data_range=data_range
    )

    # Compute the RI factor
    ri_factor = get_ri_factor(ssim_elements)
    assert np.isclose(abs(ri_factor * scaling), 1.0, atol=1e-2)


def test_global_ri_different_types(ground_truth, prediction):
    """Test that an error is raised if the two images have different types."""
    with pytest.raises(ValueError):
        get_global_ri_factor(ground_truth, [prediction])


def test_global_ri_different_shapes(ground_truth, prediction):
    """Test that an error is raised if the two images have different shapes."""
    with pytest.raises(ValueError):
        get_global_ri_factor(ground_truth, prediction[:10])


def test_global_ri_different_lengths(ground_truth, prediction):
    """Test that an error is raised if the two images have different lengths."""
    with pytest.raises(ValueError):
        get_global_ri_factor([ground_truth], [prediction, prediction])


# TODO: implement tests for global ri factor
