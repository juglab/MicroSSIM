import numpy as np
import pytest

from microssim import MicroMS3IM


def test_microms3im_class_identity(ground_truth):
    """Test the MicroMS3IM class on the identity."""
    mssim = MicroMS3IM()
    mssim.fit(ground_truth, ground_truth)

    assert np.isclose(mssim.score(ground_truth, ground_truth), 1.0)


def test_microms3im_class_similar(ground_truth, prediction):
    """Test the MicroMS3IM class on the similar images."""
    mssim = MicroMS3IM()
    mssim.fit(ground_truth, prediction)

    assert np.isclose(mssim.score(ground_truth, prediction), 1.0, atol=1e-4)


def test_microms3im_class_different(random_image, rotated_image):
    """Test the MicroMS3IM class on the vastly different images."""
    mssim = MicroMS3IM()
    mssim.fit(random_image, rotated_image)

    assert np.isclose(mssim.score(random_image, rotated_image), 0)


def test_microms3im_class_error_not_initialized(ground_truth, prediction):
    """Test that an error is raised if the class is not initialized."""
    mssim = MicroMS3IM()

    with pytest.raises(ValueError):
        mssim.score(ground_truth, prediction)


def test_microms3im_class_error_different_shapes(ground_truth, prediction):
    """Test that an error is raised if the ground truth and prediction arrays have different shapes."""
    mssim = MicroMS3IM()
    mssim.fit(ground_truth, prediction)

    with pytest.raises(ValueError):
        mssim.score(ground_truth, prediction[:-2, :-2])


def test_microms3im_class_fit_error_more_than_2d(ground_truth_stack):
    """Test that an error is raised if the ground truth and prediction have more than 3 dimensions."""
    mssim = MicroMS3IM()

    with pytest.raises(ValueError):
        mssim.score(ground_truth_stack, ground_truth_stack)
