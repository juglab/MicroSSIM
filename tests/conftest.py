import pytest
import numpy as np
from numpy.typing import NDArray

# TODO add fixtures with more dimensions (S and Z)

@pytest.fixture
def data_range() -> int:
    return 65535


@pytest.fixture
def real_scaling() -> float:
    return 10.


@pytest.fixture
def image_1(data_range) -> NDArray:
    """A random image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, data_range, (100, 100))


@pytest.fixture
def image_2(image_1, real_scaling) -> NDArray:
    """A random image similar to image_1, albeit with noise and a different 
    scaling."""
    rng = np.random.default_rng(42)
    return rng.poisson(image_1) / real_scaling

