import numpy as np

from microssim import micro_structural_similarity


def test_microssim_run(image_1, image_2):
    """Test that the metrics run."""
    micro_structural_similarity(image_1, image_2, data_range=65535)


def test_microssim_identity(image_1):
    """Test that the metrics returns 1 for the identity."""
    assert np.isclose(
        micro_structural_similarity(image_1, image_1, data_range=65535), 1.0
    )

def test_microssim_different():
    """Test that the metrics returns 0 for totally different images."""
    # create very different images
    rng = np.random.default_rng(42)
    image_1 = rng.integers(0, 65535, (100, 100)) # random image
    image_2 = np.arange(100**2).reshape(100, 100) # oredered image
    assert np.isclose(
        micro_structural_similarity(image_1, image_2, data_range=65535), 0.0, atol=1e-2
    )