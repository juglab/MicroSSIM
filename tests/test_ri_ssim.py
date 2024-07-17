import numpy as np
from skimage.metrics import structural_similarity as ssim

from ri_ssim.ri_ssim import range_invariant_structural_similarity


def test_microssim_same_as_skimage():
    gt = np.arange(10000).reshape(100, 100) / 10000
    pred = gt + np.random.normal(0, 1, gt.shape)
    ssim_skimage = ssim(gt, pred, data_range=gt.max() - gt.min(), gaussian_weights=True)
    ssim_microsim = range_invariant_structural_similarity(
        gt, pred, ri_factor=[1.0], data_range=gt.max() - gt.min(), gaussian_weights=True
    )
    assert np.isclose(ssim_skimage, ssim_microsim, atol=1e-6)
