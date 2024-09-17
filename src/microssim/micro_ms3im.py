"""MicroMS3IM metrics."""

import warnings
from typing import Optional, Union

import torch
from numpy.typing import NDArray
from torch import Tensor
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from microssim.image_processing import normalize_min_max
from microssim.micro_ssim import MicroSSIM

# TODO implement it in numpy?
# TODO return types are torch.Tensor


def micro_multiscale_structural_similarity(
    gt: Union[NDArray, list[NDArray]],
    pred: Union[NDArray, list[NDArray]],
    *,
    bg_percentile: int = 3,
    offset_gt: Optional[float] = None,
    offset_pred: Optional[float] = None,
    max_val: Optional[float] = None,
    ri_factor: Optional[float] = None,
) -> Union[float, list[float]]:
    """
    Compute the mean MicroMS3IM metric between two images.

    MicroSSIM computes a scaled version of the multiscale structural similarity index
    (MS3IM), in which images are first normalized using an offset and maximum value. A
    range-invariant factor is then estimated by maximizing a scaled MS3IM metrics
    between the normalized images.

    If the offsets are not provided, they are estimated from the images using the
    background percentile value.

    If the maximum value is not provided, it is estimated from the ground truth image
    by checking the maximum value after background subtraction.

    If the range-invariant factor is not provided, it is estimated from the normalized
    images.

    Parameters
    ----------
    gt : numpy.ndarray or list of numpy.ndarray
        Reference image.
    pred : numpy.ndarray or list of numpy.ndarray
        Image being compared to the reference.
    bg_percentile : int, default=3
        Percentile of the image considered as background.
    offset_gt : float or None, default=None
        Estimate of background pixel intensity in the reference image.
    offset_pred : float or None, default=None
        Estimate of background pixel intensity in the second image.
    max_val : float or None, default=None
        Maximum value used in normalization.
    ri_factor : float or None, default=None
        Range-invariant factor.

    Returns
    -------
    float or list of float
        Mean MicroSSIM metric between the images, either as a list if the input are
        lists or array with more than 2 dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> from microssim import micro_multiscale_structural_similarity
    >>> rng = np.random.default_rng(42)
    >>> gt = 150 + rng.integers(0, data_range, (100, 100))
    >>> pred = rng.poisson(gt) / 10. - 100
    >>> micro_multiscale_structural_similarity(gt, pred)
    """
    # generate parameters for the metrics computation
    micro_ssim = MicroMS3IM(
        bg_percentile=bg_percentile,
        offset_gt=offset_gt,
        offset_pred=offset_pred,
        max_val=max_val,
        ri_factor=ri_factor,
    )
    micro_ssim.fit(gt, pred)

    # compute the MicroMS3IM metric
    if isinstance(gt, list) or gt.ndim > 2:
        return [
            micro_ssim.score(
                gt_i,
                pred_i,
            )
            for gt_i, pred_i in zip(gt, pred)
        ]
    else:
        return micro_ssim.score(
            gt,
            pred,
        )


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

        if gt.ndim != 2:
            raise ValueError("Only 2D images are supported.")

        if ms_ssim_kwargs is None:
            ms_ssim_kwargs = {}

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
