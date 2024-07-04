"""
Multi-scale Microscopy Structural Similarity Index (MS-MicroSSIM) implementation
"""

from ._ssim_raw import structural_similarity_dict
from .ri_ssim import get_ri_factor
import numpy as np
from typing import Union
from skimage.measure import block_reduce


def micro_MS_SSIM(
    target_img,
    pred_img,
    *,
    betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
    win_size=None,
    data_range=None,
    channel_axis=None,
    gaussian_weights=False,
    ri_factor: Union[float, None] = None,
    return_individual_components: bool = False,
    **kwargs,
):
    mcs_list = []
    for _ in range(len(betas)):
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
            ri_factor = get_ri_factor(ssim_dict)

        ux, uy, vx, vy, vxy, C1, C2 = (
            ssim_dict["ux"],
            ssim_dict["uy"],
            ssim_dict["vx"],
            ssim_dict["vy"],
            ssim_dict["vxy"],
            ssim_dict["C1"],
            ssim_dict["C2"],
        )
        A1, A2, B1, B2 = (
            2 * ri_factor * ux * uy + C1,
            2 * ri_factor * vxy + C2,
            ux**2 + (ri_factor**2) * uy**2 + C1,
            vx + (ri_factor**2) * vy + C2,
        )
        assert A1.shape == A2.shape == B1.shape == B2.shape
        assert len(A1.shape) == 2
        sim = (A1 / B1).mean()
        contrast_sensitivity = (A2 / B2).mean()

        mcs_list.append(contrast_sensitivity)

        pred_img = block_reduce(pred_img, (2, 2), np.mean)
        target_img = block_reduce(target_img, (2, 2), np.mean)

    mcs_list[-1] = sim
    mcs_stack = np.stack(mcs_list)

    betas = np.array(betas).reshape(-1, 1)
    mcs_weighted = mcs_stack**betas
    return np.prod(mcs_weighted, axis=0)
