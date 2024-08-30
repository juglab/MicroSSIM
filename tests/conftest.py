import pytest
import numpy as np


@pytest.fixture
def image_1():
    """A random image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 65535, (100, 100))


@pytest.fixture
def image_2(image_1):
    """A random image similar to image_1, albeit with noise and a different scaling."""
    rng = np.random.default_rng(42)
    return 0.1 * rng.poisson(image_1)
