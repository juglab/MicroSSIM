import numpy as np
import pytest

from microssim import MicroMS3IM, micro_multiscale_structural_similarity


def test_microms3im_function_identity(ground_truth):
    """Test the micro ms structural similarity function on the identity."""
    assert np.isclose(
        micro_multiscale_structural_similarity(ground_truth, ground_truth), 1.0
    )


def test_microms3im_function_lists(ground_truth_list):
    """Test that the micro ms structural similarity function returns the correct number of values."""
    assert len(
        micro_multiscale_structural_similarity(ground_truth_list, ground_truth_list)
    ) == len(ground_truth_list)


def test_microms3im_function_identity_lists(ground_truth_list):
    """Test the micro structural similarity function on identical lists."""
    assert np.allclose(
        micro_multiscale_structural_similarity(ground_truth_list, ground_truth_list),
        1.0,
    )


def test_microms3im_function_identity_stacks(ground_truth_stack):
    """Test the micro structural similarity function on identical stacks."""
    assert np.allclose(
        micro_multiscale_structural_similarity(ground_truth_stack, ground_truth_stack),
        1.0,
    )


def test_microms3im_function_similar(ground_truth, prediction):
    """Test the micro structural similarity function on similar arrays."""
    assert np.isclose(
        micro_multiscale_structural_similarity(ground_truth, prediction), 1.0, atol=1e-4
    )


def test_microms3im_function_similar_lists(ground_truth_list, prediction_list):
    """Test the micro structural similarity function on similar list of arrays."""
    assert np.allclose(
        micro_multiscale_structural_similarity(ground_truth_list, prediction_list),
        1.0,
        atol=1e-4,
    )


def test_microms3im_function_similar_stacks(ground_truth_stack, prediction_stack):
    """Test the micro structural similarity function on similar arrays."""
    assert np.allclose(
        micro_multiscale_structural_similarity(ground_truth_stack, prediction_stack),
        1.0,
        atol=1e-4,
    )


def test_microms3im_function_different(random_image, rotated_image):
    """Test that the metrics returns 0 for totally different images."""
    assert np.isclose(
        micro_multiscale_structural_similarity(random_image, rotated_image), 0.0
    )


def test_microms3im_function_different_lists(random_image_list, rotated_image_list):
    """Test that the metrics returns 0 for totally different image lists."""
    assert np.allclose(
        micro_multiscale_structural_similarity(random_image_list, rotated_image_list),
        0.0,
    )


def test_microms3im_function_different_stacks(random_image_stack, rotated_image_stack):
    """Test that the metrics returns 0 for totally different image stacks."""
    assert np.allclose(
        micro_multiscale_structural_similarity(random_image_stack, rotated_image_stack),
        0.0,
        atol=1e-4,
    )


def test_microms3im_class_identity(ground_truth):
    """Test the MicroMS3IM class on the identity."""
    ms3im = MicroMS3IM()
    ms3im.fit(ground_truth, ground_truth)

    assert np.isclose(ms3im.score(ground_truth, ground_truth), 1.0)


def test_microms3im_class_similar(ground_truth, prediction):
    """Test the MicroMS3IM class on the similar images."""
    ms3im = MicroMS3IM()
    ms3im.fit(ground_truth, prediction)

    assert np.isclose(ms3im.score(ground_truth, prediction), 1.0, atol=1e-4)


def test_microms3im_class_different(random_image, rotated_image):
    """Test the MicroMS3IM class on the vastly different images."""
    ms3im = MicroMS3IM()
    ms3im.fit(random_image, rotated_image)

    assert np.isclose(ms3im.score(random_image, rotated_image), 0)


def test_microms3im_class_error_not_initialized(ground_truth, prediction):
    """Test that an error is raised if the class is not initialized."""
    ms3im = MicroMS3IM()

    with pytest.raises(ValueError):
        ms3im.score(ground_truth, prediction)


def test_microms3im_class_error_different_shapes(ground_truth, prediction):
    """Test that an error is raised if the ground truth and prediction arrays have different shapes."""
    ms3im = MicroMS3IM()
    ms3im.fit(ground_truth, prediction)

    with pytest.raises(ValueError):
        ms3im.score(ground_truth, prediction[:-2, :-2])


def test_microms3im_class_fit_error_more_than_2d(ground_truth_stack):
    """Test that an error is raised if the ground truth and prediction have more than 3 dimensions."""
    ms3im = MicroMS3IM()

    with pytest.raises(ValueError):
        ms3im.score(ground_truth_stack, ground_truth_stack)
