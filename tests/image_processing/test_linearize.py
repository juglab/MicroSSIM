import numpy as np

from microssim.image_processing import linearize


def test_linearize_list():
    """Test the linearize_list function."""
    l = [i * np.ones((i, i)) for i in range(1, 4)]

    result = linearize.linearize_list(l)
    assert len(result.shape) == 1
    assert result.shape[0] == np.sum([i * i for i in range(1, 4)])
    assert len(set(result)) == 3
