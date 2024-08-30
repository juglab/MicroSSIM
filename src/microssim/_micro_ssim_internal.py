import numpy as np

from microssim import micro_structural_similarity

from .mse_ri_ssim import get_mse_based_factor


def mse_based_range_invariant_structural_similarity(
    target_img,
    pred_img,
    *,
    win_size=None,
    data_range=None,
    channel_axis=None,
    gaussian_weights=False,
    return_individual_components=False,
    **kwargs,
):
    ri_factor = micro_structural_similarity(target_img[None], pred_img[None])

    return micro_structural_similarity(
        target_img,
        pred_img,
        win_size=win_size,
        data_range=data_range,
        channel_axis=channel_axis,
        gaussian_weights=gaussian_weights,
        ri_factor=ri_factor,
        return_individual_components=return_individual_components,
        **kwargs,
    )


def remove_background(x, pmin=3, dtype=np.float32):
    mi = np.percentile(x, pmin, keepdims=True)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        x = x - mi
    return x
