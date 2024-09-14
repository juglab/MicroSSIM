import numpy as np

from microssim import MicroMS3IM


def test_MicroMS3IM_class_identity(ground_truth):
    """Test the MicroMS3IM class on the identity."""
    mssim = MicroMS3IM()
    mssim.fit(ground_truth, ground_truth)

    assert np.isclose(mssim.score(ground_truth, ground_truth), 1.0)


def test_MicroMS3IM_class_similar(ground_truth, prediction):
    """Test the MicroMS3IM class on the similar images."""
    mssim = MicroMS3IM()
    mssim.fit(ground_truth, prediction)

    assert np.isclose(mssim.score(ground_truth, prediction), 1.0, atol=1e-4)


def test_MicroMS3IM_class_different(random_image, rotated_image):
    """Test the MicroMS3IM class on the vastly different images."""
    mssim = MicroMS3IM()
    mssim.fit(random_image, rotated_image)

    assert np.isclose(mssim.score(random_image, rotated_image), 0)
