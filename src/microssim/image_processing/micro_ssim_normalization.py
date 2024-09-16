"""MicroSSIM normalization functions."""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


# TODO add tests
def normalize_min_max(
    images: Union[list[NDArray], NDArray], min_val: float, max_val: float
) -> Union[list[NDArray], NDArray]:
    """
    Normalize the images using provided minimum and maximum values.

    Parameters
    ----------
    images : numpy.ndarray or list of numpy.ndarray
        Image or list of images to normalize.
    min_val : float
        Minimum value used in normalization.
    max_val : float
        Maximum value used in normalization.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        Normalized image or list of normalized images.
    """
    if isinstance(images, list):
        return [normalize_min_max(x, min_val, max_val) for x in images]

    return (images - min_val) / max_val


def compute_norm_parameters(
    gt: NDArray,
    pred: NDArray,
    bg_percentile: float = 3.0,
    offset_gt: Optional[float] = None,
    offset_pred: Optional[float] = None,
    max_val: Optional[float] = None,
) -> tuple[float, float, float]:
    """
    Compute the parameters used to normalize the images for MicroSSIM.

    If the offsets are provided, they are simply returned. Otherwise, they are
    estimated from the images using the background percentile value.

    If the maximum value is provided, it is simply returned. Otherwise, it is
    estimated from the ground truth image by checking the maximum value after
    background subtraction.

    Parameters
    ----------
    gt : NDArray
        Reference image.
    pred : NDArray
        Image being compared to the reference.
    bg_percentile : float, default=3
        Percentile of the image considered as background.
    offset_gt : float or None, default=None
        Estimate of background pixel intensity in the reference image.
    offset_pred : float or None, default=None
        Estimate of background pixel intensity in the second image.
    max_val : float or None, default=None
        Maximum value used in normalization.

    Returns
    -------
    (float, float, float)
        A tuple containing the ground-truth offset, the prediction offset and the
        maximum value.
    """
    if offset_gt is None:
        offset_gt = np.percentile(gt, bg_percentile, keepdims=False)

    if offset_pred is None:
        offset_pred = np.percentile(pred, bg_percentile, keepdims=False)

    if max_val is None:
        max_val = (gt - offset_gt).max()

    return offset_gt, offset_pred, max_val
