from _ssim_raw import structural_similarity_dict
from _mse_ri_factor import get_mse_based_factor
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Union


def _ssim_from_params(alpha, ux, uy, vx, vy, vxy, C1, C2):
    A1, A2, B1, B2 = (
        2 * alpha * ux * uy + C1,
        2 * alpha * vxy + C2,
        ux**2 + (alpha**2) * uy**2 + C1,
        vx + (alpha**2) * vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D
    return np.mean(S)


def _get_ri_factor(ssim_dict: Dict[str, np.ndarray]):
    other_args = (
        ssim_dict["ux"],
        ssim_dict["uy"],
        ssim_dict["vx"],
        ssim_dict["vy"],
        ssim_dict["vxy"],
        ssim_dict["C1"],
        ssim_dict["C2"],
    )
    initial_guess = np.array([1])
    res = minimize(
        lambda *args: -1 * _ssim_from_params(*args), initial_guess, args=other_args
    )
    return res.x[0]


def mse_based_range_invariant_structural_similarity(
    target_img,
    pred_img,
    *,
    win_size=None,
    data_range=None,
    channel_axis=None,
    gaussian_weights=False,
    **kwargs,
):
    ri_factor = get_mse_based_factor(target_img[None], pred_img[None])

    return range_invariant_structural_similarity(
        target_img,
        pred_img,
        win_size=win_size,
        data_range=data_range,
        channel_axis=channel_axis,
        gaussian_weights=gaussian_weights,
        ri_factor=ri_factor,
        **kwargs,
    )


def range_invariant_structural_similarity(
    target_img,
    pred_img,
    *,
    win_size=None,
    data_range=None,
    channel_axis=None,
    gaussian_weights=False,
    ri_factor: Union[float, None] = None,
    **kwargs,
):
    ssim_dict = structural_similarity_dict(
        target_img,
        pred_img,
        win_size=win_size,
        data_range=data_range,
        channel_axis=channel_axis,
        gaussian_weights=gaussian_weights,
        **kwargs,
    )
    if ri_factor is None:
        ri_factor = _get_ri_factor(ssim_dict)
    ux, uy, vx, vy, vxy, C1, C2 = (
        ssim_dict["ux"],
        ssim_dict["uy"],
        ssim_dict["vx"],
        ssim_dict["vy"],
        ssim_dict["vxy"],
        ssim_dict["C1"],
        ssim_dict["C2"],
    )

    return _ssim_from_params(ri_factor, ux, uy, vx, vy, vxy, C1, C2)


if __name__ == "__main__":
    import numpy as np
    from skimage.io import imread

    def load_tiff(path):
        """
        Returns a 4d numpy array: num_imgs*h*w*num_channels
        """
        data = imread(path, plugin="tifffile")
        return data

    img1 = load_tiff(
        "/group/jug/ashesh/data/paper_stats/Test_P64_G32_M50_Sk8/gt_D21.tif"
    )
    img2 = load_tiff(
        "/group/jug/ashesh/data/paper_stats/Test_P64_G32_M50_Sk8/pred_training_disentangle_2404_D21-M3-S0-L8_1.tif"
    )
    ch_idx = 0
    img_gt = img1[0, ..., ch_idx]
    img_pred = img2[0, ..., ch_idx]
    print(
        "SSIM",
        range_invariant_structural_similarity(
            img_gt,
            img_pred,
            data_range=img_gt.max() - img_gt.min(),
            ri_factor=1.0,
        ),
    )

    print(
        "RI-SSIM",
        range_invariant_structural_similarity(
            img_gt,
            img_pred,
            data_range=img_gt.max() - img_gt.min(),
        ),
    )

    print(
        "RI-SSIM using MSE based:",
        mse_based_range_invariant_structural_similarity(
            img_gt,
            img_pred,
            data_range=img_gt.max() - img_gt.min(),
        ),
    )
