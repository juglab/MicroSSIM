"""Utility functions for calculating a rescaling factor based on the mean squared error."""

import numpy as np
from numpy.typing import NDArray


def _get_factor(gt: NDArray, pred: NDArray) -> NDArray:
    """
    Compute MSE rescaling factor.

    Parameters
    ----------
    gt : NDArray
        Reference image.
    pred : NDArray
        Image to be compared with the reference image.

    Returns
    -------
    NDArray
        MSE-based rescaling factor along the first dimension of the reference array.
    """
    factor = np.sum(gt * pred, axis=1, keepdims=True) / (
        np.sum(pred * pred, axis=1, keepdims=True)
    )
    return factor


def get_mse_ri_factor(gt: NDArray, pred: NDArray) -> NDArray:
    """
    Compute MSE-based rescaling factor.

    The MSE-based rescaling factor is used to rescale the prediction to the same
    range of values as the reference image. Adapted from
    https://github.com/juglab/ScaleInvPSNR.

    Parameters
    ----------
    gt : NDArray
        Reference image.
    pred : NDArray
        Image to be compared with the reference image.

    Returns
    -------
    NDArray
        MSE-based rescaling factor.

    Raises
    ------
    ValueError
        If the input arrays do not have the same shape.
    """
    if gt.shape != pred.shape:
        raise ValueError("gt and pred must have the same shape.")

    # flatten extra dimensions into a 2D array
    gt = gt.reshape(len(gt), -1)
    pred = pred.reshape(len(gt), -1)

    return _get_factor(gt, pred)
