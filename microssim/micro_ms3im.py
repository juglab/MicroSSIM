from typing import Union

import numpy as np
import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from microssim.micro_ssim import MicroSSIM


class MicroMS3IM(MicroSSIM):
    """
    Extension of MS-SSIM for microscopy data.
    """

    def score(
        self,
        gt: np.ndarray,
        pred: np.ndarray,
        return_individual_components: bool = False,
        ms_ssim_kwargs: Union[dict, None] = None,
    ):
        if ms_ssim_kwargs is None:
            ms_ssim_kwargs = {}

        if not self._initialized:
            raise ValueError(
                "fit method was not called before score method. Expected behaviour is to call fit \
                  with ALL DATA and then call score(), with individual images.\
                  Using all data for fitting ensures better estimation of ri_factor."
            )
        assert (
            gt.shape == pred.shape
        ), "Groundtruth and prediction must have same shape."
        assert len(gt.shape) == 2, "Only 2D images are supported."

        gt_norm = self.normalize_gt(gt)
        pred_norm = self.normalize_prediction(pred)

        gt_torch = torch.Tensor(gt_norm[None, None])
        pred_torch = torch.Tensor(pred_norm[None, None])
        pred_torch = pred_torch * self._ri_factor
        ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=gt_norm.max() - gt_norm.min(), **ms_ssim_kwargs
        )
        return ms_ssim(pred_torch, gt_torch)
