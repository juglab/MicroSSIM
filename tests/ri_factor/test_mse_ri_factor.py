import numpy as np

from microssim.ri_factor import get_mse_ri_factor


def test_identity(ground_truth):
    """Test that the MSE RI factor is 1 when the images are the same."""
    factor = get_mse_ri_factor(ground_truth, ground_truth)
    assert np.allclose(factor, 1.0)


def test_different_images(ground_truth, prediction, real_scaling):
    """Test that the MSE RI is close to the real scaling."""
    factor = get_mse_ri_factor(ground_truth, prediction)
    assert np.allclose(factor, real_scaling, atol=0.1)
