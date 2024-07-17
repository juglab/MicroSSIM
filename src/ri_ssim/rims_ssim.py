"""
Multi-scale Microscopy Structural Similarity Index (MS-MicroSSIM) implementation
"""

from typing import Tuple, Union

# TODO can we remove dependency on torch?
import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from .ri_ssim import get_ri_factor
from .ssim import compute_ssim_elements


def range_invariant_multiscale_structural_similarity(
    target_img,
    pred_img,
    *,
    betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
    win_size=None,
    data_range=None,
    channel_axis=None,
    gaussian_weights=False,
    ri_factor: Union[float, None] = None,
    return_ri_factor=False,
    **kwargs,
) -> Union[float, Tuple[float, float]]:
    if ri_factor is None:
        ri_factor_kwargs = kwargs.get("ri_factor_kwargs", {})

        ssim_dict = compute_ssim_elements(
            target_img,
            pred_img,
            win_size=win_size,
            data_range=data_range,
            channel_axis=channel_axis,
            gaussian_weights=gaussian_weights,
            **ri_factor_kwargs,
        )
        ri_factor = get_ri_factor(ssim_dict)
    pred_img = pred_img * ri_factor

    gt_torch = torch.Tensor(target_img[None, None] * 1.0)
    pred_torch = torch.Tensor(pred_img[None, None] * 1.0)
    ms_ssim_kwargs = kwargs.get("ms_ssim_kwargs", {})
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=data_range,
        gaussian_kernel=gaussian_weights,
        betas=betas,
        **ms_ssim_kwargs,
    )

    if return_ri_factor:
        return ms_ssim(pred_torch, gt_torch), ri_factor

    return ms_ssim(pred_torch, gt_torch)
