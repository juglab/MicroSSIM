import numpy as np
import pytest
from numpy.typing import NDArray

# TODO add fixtures with more dimensions (S and Z)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def data_range() -> int:
    return 65535


@pytest.fixture
def real_scaling() -> float:
    return 10.0


@pytest.fixture
def ground_truth(rng: np.random.Generator, data_range) -> NDArray:
    """A random image.

    Note: torch MS-SSIM requires images of shape larger than (160, 160) after the
    convolution.
    """
    return rng.integers(0, data_range, (256, 256))


@pytest.fixture
def prediction(rng: np.random.Generator, ground_truth, real_scaling) -> NDArray:
    """An image similar to the ground_truth, albeit with noise and a different
    scaling."""
    return rng.poisson(ground_truth) / real_scaling


@pytest.fixture
def ground_truth_list(rng: np.random.Generator, data_range) -> list[NDArray]:
    """A list of random images with different sizes.

    Note: torch MS-SSIM requires images of shape larger than (160, 160) after the
    convolution.
    """
    l = []
    for i in range(25, 28):
        l.append(rng.integers(0, data_range, (10 * i, 10 * i)))

    return l


@pytest.fixture
def prediction_list(
    rng: np.random.Generator, ground_truth_list, real_scaling
) -> list[NDArray]:
    """A list of images similar to the ground_truth_list, albeit with noise and a
    different scaling."""
    return [rng.poisson(g_i) / real_scaling for g_i in ground_truth_list]


@pytest.fixture
def ground_truth_stack(rng: np.random.Generator, data_range) -> NDArray:
    """A stack of random images.

    Note: torch MS-SSIM requires images of shape larger than (160, 160) after the
    convolution.
    """
    return rng.integers(0, data_range, (9, 256, 256))


@pytest.fixture
def prediction_stack(
    rng: np.random.Generator, ground_truth_stack, real_scaling
) -> NDArray:
    """A stack of images similar to the ground_truth_stack, albeit with noise and a
    different scaling."""
    return rng.poisson(ground_truth_stack) / real_scaling


@pytest.fixture
def random_image(rng: np.random.Generator, data_range) -> NDArray:
    """A random image.

    Note: torch MS-SSIM requires images of shape larger than (160, 160) after the
    convolution.
    """
    return 200 + rng.integers(0, data_range - 200, (256, 256))


@pytest.fixture
def rotated_image(random_image) -> NDArray:
    """Rotated image."""
    return np.rot90(random_image**2)


@pytest.fixture
def random_image_list(rng: np.random.Generator, data_range) -> list[NDArray]:
    """A list of random images.

    Note: torch MS-SSIM requires images of shape larger than (160, 160) after the
    convolution.
    """
    l = []
    for i in range(25, 28):
        l.append(200 + rng.integers(0, data_range - 200, (10 * i, 10 * i)))

    return l


@pytest.fixture
def rotated_image_list(random_image_list) -> list[NDArray]:
    """Rotated list of images."""
    return [np.rot90(r_i**2) for r_i in random_image_list]


@pytest.fixture
def random_image_stack(rng: np.random.Generator, data_range) -> NDArray:
    """A stack of random images.

    Note: torch MS-SSIM requires images of shape larger than (160, 160) after the
    convolution.
    """
    return 200 + rng.integers(0, data_range - 200, (9, 256, 256))


@pytest.fixture
def rotated_image_stack(random_image_stack) -> NDArray:
    """Rotated stack of images."""
    return np.rot90(random_image_stack**2, axes=(1, 2))
