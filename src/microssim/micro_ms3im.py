"""MicroMS3IM metrics."""

import warnings

import torch
from numpy.typing import NDArray
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from microssim.image_processing import normalize_min_max
from microssim.micro_ssim import MicroSSIM


class MicroMS3IM(MicroSSIM):
    """
    A class computing the MicroMS3IM metric between images.

    In addition to computing the metrics, this class allows to inspect the parameters
    estimated along the way, such as the offsets, the max value and the range-invariant
    factor.

    Attributes
    ----------
    _bg_percentile : int
        Percentile of the image considered as background.
    _offset_pred : float
        Estimate of background pixel intensity in the prediction image.
    _offset_gt : float
        Estimate of background pixel intensity in the ground truth image.
    _max_val : float
        Maximum value used in normalization.
    _ri_factor : float
        MicroSSIM scaling factor.
    _initialized : bool
        Whether the class has been initialized and can compute scores between images.
    """

    def score(
        self,
        gt: NDArray,
        pred: NDArray,
        return_individual_components: bool = False,
        **ms_ssim_kwargs,
    ) -> float:
        """Compute the metrics between two arrays.

        Parameters
        ----------
        gt : numpy.ndarray
            Reference array.
        pred : NDArray
            Array to be compared to the reference array.
        return_individual_components : bool, default=False
            Unused argument.
        **ms_ssim_kwargs : dict
            Additional keyword arguments to be passed to the
            `torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure` class.

        Returns
        -------
        float or ScaledSSIM
            MicroSSIM metric between the arrays.

        Raises
        ------
        ValueError
            If the `fit` method has not been called before calling this method.
        ValueError
            If the groundtruth and prediction arrays have different shapes.
        ValueError
            If the arrays are not 2D.
        """
        if return_individual_components:
            warnings.warn(
                "The `return_individual_components` argument is not supported for "
                "the MS-SSIM metric. Ignoring it."
            )

        if not self._initialized:
            raise ValueError(
                "MicroSSIM was not initialized, call the `fit` method first. It is "
                "advised to run the `fit` method on entire datasets rather than on "
                "pairs of images."
            )

        if gt.shape != pred.shape:
            raise ValueError("Groundtruth and prediction must have the same shape.")

        if gt.ndim < 2 or gt.ndim > 2:
            raise ValueError("Only 2D images are supported.")

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

        # normalize the images
        gt_norm = normalize_min_max(gt, self._offset_gt, self._max_val)
        pred_norm = normalize_min_max(pred, self._offset_pred, self._max_val)

        # convert to torch tensors
        gt_torch = torch.Tensor(gt_norm[None, None])
        pred_torch = torch.Tensor(pred_norm[None, None])

        # rescale
        pred_torch = pred_torch * self._ri_factor

        # compute the MS-SSIM
        ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=gt_norm.max() - gt_norm.min(), **ms_ssim_kwargs
        )

        return ms_ssim(pred_torch, gt_torch)
