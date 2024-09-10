import numpy as np

from microssim.ri_factor import get_mse_ri_factor


def test_identity(image_1):
    """Test that the MSE RI factor is 1 when the images are the same."""
    factor = get_mse_ri_factor(image_1, image_1)
    assert np.allclose(factor, 1.0)


def test_different_images(image_1, image_2, real_scaling):
    """Test that the MSE RI is close to the real scaling."""
    factor = get_mse_ri_factor(image_1, image_2)
    assert np.allclose(factor, real_scaling, atol = 0.1)
