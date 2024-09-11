import numpy as np
import pytest
from numpy.typing import NDArray

# TODO add fixtures with more dimensions (S and Z)


@pytest.fixture
def data_range() -> int:
    return 65535


@pytest.fixture
def real_scaling() -> float:
    return 10.0


@pytest.fixture
def ground_truth(data_range) -> NDArray:
    """A random image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, data_range, (100, 100))


@pytest.fixture
def prediction(ground_truth, real_scaling) -> NDArray:
    """A random image similar to image_1, albeit with noise and a different
    scaling."""
    rng = np.random.default_rng(42)
    return rng.poisson(ground_truth) / real_scaling


@pytest.fixture
def random_image(data_range) -> NDArray:
    """A random image."""
    rng = np.random.default_rng(42)
    return 200 + rng.integers(0, data_range - 200, (8, 8))


@pytest.fixture
def ordered_image() -> NDArray:
    return np.arange(8**2).reshape(8, 8)
