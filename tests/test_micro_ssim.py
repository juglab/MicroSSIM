import numpy as np

from microssim.micro_ssim import (
    MicroSSIM,
    _compute_micro_ssim,
    micro_structural_similarity,
)


def test_microssim_identity(ground_truth):
    """Test that the metrics returns 1 for the identity."""
    assert np.isclose(
        _compute_micro_ssim(ground_truth, ground_truth, data_range=65535), 1.0
    )


def test_microssim_different():
    """Test that the metrics returns 0 for totally different images."""
    # create very different images
    rng = np.random.default_rng(42)
    image_1 = rng.integers(0, 65535, (100, 100))  # random image
    image_2 = np.arange(100**2).reshape(100, 100)  # ordered image
    assert np.isclose(
        _compute_micro_ssim(image_1, image_2, data_range=65535), 0.0, atol=1e-2
    )


def test_micro_structural_similarity_identity(ground_truth):
    """Test the micro structural similarity function on the identity."""
    assert np.isclose(micro_structural_similarity(ground_truth, ground_truth), 1.0)


def test_micro_structural_similarity_identity_list(ground_truth):
    """Test the micro structural similarity function on the identity."""
    l = [
        ground_truth,
        ground_truth - 5,
        ground_truth - 50,
        ground_truth - 15,
        ground_truth - 42,
    ]

    assert np.allclose(micro_structural_similarity(l, l), 1.0)


def test_micro_structural_similarity_identity_stack(ground_truth):
    """Test the micro structural similarity function on the identity."""
    l = np.stack(
        [
            ground_truth,
            ground_truth - 5,
            ground_truth - 50,
            ground_truth - 15,
            ground_truth - 42,
        ],
        axis=0,
    )

    assert np.allclose(micro_structural_similarity(l, l), 1.0)


def test_micro_structural_similarity_similar_arrays(ground_truth, prediction):
    """Test the micro structural similarity function on different arrays."""
    assert np.isclose(
        micro_structural_similarity(ground_truth, prediction), 1.0, atol=1e-2
    )


def test_micro_structural_similarity_different_arrays_0(random_image, ordered_image):
    """Test that the metrics returns 0 for totally different images."""
    assert np.isclose(
        micro_structural_similarity(random_image, ordered_image), 0.0, atol=1e-2
    )


def test_micro_structural_similarity_different_arrays_0_list():
    """Test that the metrics returns 0 for totally different images."""
    # create very different images
    rng = np.random.default_rng(42)

    l1 = []
    l2 = []
    for i in range(3, 8):
        l1.append(200 + rng.integers(0, 50_000, (i * 10, i * 10)))  # random image
        l2.append(np.arange((i * 10) ** 2).reshape((i * 10, i * 10)))  # ordered image

    assert np.allclose(micro_structural_similarity(l1, l2), 0.0, atol=1e-2)


def test_microssim_identity(ground_truth):
    """Test the MicroSSIM class on the identity."""
    mssim = MicroSSIM()
    mssim.fit(ground_truth, ground_truth)

    assert np.isclose(mssim.score(ground_truth, ground_truth), 1.0)


def test_microssim_identity_list(ground_truth):
    """Test the MicroSSIM class on the identity."""
    l = [
        ground_truth,
        ground_truth - 5,
        ground_truth - 50,
        ground_truth - 15,
        ground_truth - 42,
    ]

    mssim = MicroSSIM()
    mssim.fit(l, l)

    for g, p in zip(l, l):
        assert np.isclose(mssim.score(g, p), 1.0)


def test_microssim_identity_stack(ground_truth):
    """Test the MicroSSIM class on the identity."""
    l = np.stack(
        [
            ground_truth,
            ground_truth - 5,
            ground_truth - 50,
            ground_truth - 15,
            ground_truth - 42,
        ],
        axis=0,
    )

    mssim = MicroSSIM()
    mssim.fit(l, l)

    for g, p in zip(l, l):
        assert np.isclose(mssim.score(g, p), 1.0)


def test_microssim_different_arrays_0(random_image, ordered_image):
    """Test that the metrics returns 0 for totally different images."""
    mssim = MicroSSIM()
    mssim.fit(random_image, ordered_image)

    assert np.isclose(mssim.score(random_image, ordered_image), 0.0, atol=1e-2)


def test_agreement_similar_images(ground_truth, prediction):
    """Test that micro_structural_similarity and MicroSSIM agree."""
    assert np.isclose(
        micro_structural_similarity(ground_truth, prediction),
        MicroSSIM().fit(ground_truth, prediction).score(ground_truth, prediction),
    )


# TODO why is the agreement not perfect?
def test_agreement_different_images(random_image, ordered_image):
    """Test that micro_structural_similarity and MicroSSIM agree."""
    assert np.isclose(
        micro_structural_similarity(random_image, ordered_image),
        MicroSSIM().fit(random_image, ordered_image).score(random_image, ordered_image),
        atol=1e-3,
    )
