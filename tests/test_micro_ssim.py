import numpy as np

from microssim.micro_ssim import (
    MicroSSIM,
    _compute_micro_ssim,
    micro_structural_similarity,
)


def test_compute_microssim_identity(ground_truth, data_range):
    """Test that the metrics returns 1 for the identity."""
    assert np.isclose(
        _compute_micro_ssim(ground_truth, ground_truth, data_range=data_range), 1.0
    )


def test_compute_microssim_similar(ground_truth, prediction, data_range):
    """Test that the metrics returns close to 1 for similar images."""
    assert np.isclose(
        _compute_micro_ssim(ground_truth, prediction, data_range=data_range),
        1.0,
        atol=1e-4,
    )


def test_compute_microssim_different(random_image, rotated_image, data_range):
    """Test that the metrics returns 0 for totally different images."""
    assert np.isclose(
        _compute_micro_ssim(random_image, rotated_image**2, data_range=data_range), 0.0
    )


def test_microssim_function_identity(ground_truth):
    """Test the micro structural similarity function on the identity."""
    assert np.isclose(micro_structural_similarity(ground_truth, ground_truth), 1.0)


def test_microssim_function_identity_lists(ground_truth_list):
    """Test the micro structural similarity function on the identity."""
    assert np.allclose(
        micro_structural_similarity(ground_truth_list, ground_truth_list), 1.0
    )


def test_microssim_function_identity_stacks(ground_truth_stack):
    """Test the micro structural similarity function on the identity."""
    assert np.allclose(
        micro_structural_similarity(ground_truth_stack, ground_truth_stack), 1.0
    )


def test_microssim_function_similar(ground_truth, prediction):
    """Test the micro structural similarity function on similar arrays."""
    assert np.isclose(
        micro_structural_similarity(ground_truth, prediction), 1.0, atol=1e-4
    )


def test_microssim_function_similar_lists(ground_truth_list, prediction_list):
    """Test the micro structural similarity function on similar list of arrays."""
    assert np.allclose(
        micro_structural_similarity(ground_truth_list, prediction_list), 1.0, atol=1e-4
    )


def test_microssim_function_similar_stacks(ground_truth_stack, prediction_stack):
    """Test the micro structural similarity function on similar arrays."""
    assert np.allclose(
        micro_structural_similarity(ground_truth_stack, prediction_stack),
        1.0,
        atol=1e-4,
    )


def test_microssim_function_different(random_image, rotated_image):
    """Test that the metrics returns 0 for totally different images."""
    assert np.isclose(micro_structural_similarity(random_image, rotated_image), 0.0)


def test_microssim_function_different_lists(random_image_list, rotated_image_list):
    """Test that the metrics returns 0 for totally different image lists."""
    assert np.allclose(
        micro_structural_similarity(random_image_list, rotated_image_list), 0.0
    )


def test_microssim_function_different_stacks(random_image_stack, rotated_image_stack):
    """Test that the metrics returns 0 for totally different image stacks."""
    assert np.allclose(
        micro_structural_similarity(random_image_stack, rotated_image_stack), 0.0
    )


def test_microssim_class_identity(ground_truth):
    """Test the MicroSSIM class on the identity."""
    mssim = MicroSSIM()
    mssim.fit(ground_truth, ground_truth)

    assert np.isclose(mssim.score(ground_truth, ground_truth), 1.0)


def test_microssim_class_identity_stack(ground_truth_stack):
    """Test the MicroSSIM class on the identity."""
    mssim = MicroSSIM()
    mssim.fit(ground_truth_stack, ground_truth_stack)

    assert np.isclose(mssim.score(ground_truth_stack, ground_truth_stack), 1.0)


def test_microssim_class_similar(ground_truth, prediction):
    """Test the MicroSSIM class on the similar images."""
    mssim = MicroSSIM()
    mssim.fit(ground_truth, prediction)

    assert np.isclose(mssim.score(ground_truth, prediction), 1.0, atol=1e-4)


def test_microssim_class_similar_stacks(ground_truth_stack, prediction_stack):
    """Test the MicroSSIM class on the identical stacks."""
    mssim = MicroSSIM()
    mssim.fit(ground_truth_stack, prediction_stack)

    assert np.isclose(mssim.score(ground_truth_stack, prediction_stack), 1.0, atol=1e-4)


def test_microssim_class_different(random_image, rotated_image):
    """Test the MicroSSIM class on the vastly different images."""
    mssim = MicroSSIM()
    mssim.fit(random_image, rotated_image)

    assert np.isclose(mssim.score(random_image, rotated_image), 0)


def test_microssim_class_different_stacks(random_image_stack, rotated_image_stack):
    """Test the MicroSSIM class on the identical stacks."""
    mssim = MicroSSIM()
    mssim.fit(random_image_stack, rotated_image_stack)

    assert np.isclose(mssim.score(random_image_stack, rotated_image_stack), 0)


def test_agreement_similar_images(ground_truth, prediction):
    """Test that micro_structural_similarity and MicroSSIM agree."""
    assert np.isclose(
        micro_structural_similarity(ground_truth, prediction),
        MicroSSIM().fit(ground_truth, prediction).score(ground_truth, prediction),
    )


def test_agreement_different_images(random_image, rotated_image):
    """Test that micro_structural_similarity and MicroSSIM agree."""
    assert np.isclose(
        micro_structural_similarity(random_image, rotated_image),
        MicroSSIM().fit(random_image, rotated_image).score(random_image, rotated_image),
    )
