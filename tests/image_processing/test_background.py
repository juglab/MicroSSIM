import numpy as np

from microssim.image_processing import get_background, remove_background


def test_get_background():
    """Test that the percentile is correctly calculated."""
    x = np.arange(100).reshape(10, 10)

    # compute percentile
    val = get_background(x, 10)
    assert np.isclose(val, 9.9)


def test_remove_background():
    """Test that the background is correctly removed."""
    x = np.arange(100).reshape(10, 10).astype(np.float32)

    # remove background
    y = remove_background(x, 10)
    val = get_background(x, 10)

    assert np.allclose(y, x - val)
