"""Normalization as done in the CSBDEEP library and CARE approach.

Code adapted from https://github.com/CSBDeep/CSBDeep under BSD-3-Clause license.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def normalize_care(
    x: NDArray,
    pmin: float = 3,
    pmax: float = 99.8,
    axis: Optional[int] = None,
    eps: float = 1e-20,
    dtype: type = np.float32,
) -> NDArray:
    """
    Normalize input image using min and max percentiles.

    This approach was used in the CARE method.

    Parameters
    ----------
    x : NDArray
        Input image.
    pmin : float, default=3
        Lower percentile.
    pmax : float, default=99.8
        Upper percentile.
    axis : int or None, default=None
        Axis or axes along which to compute the percentiles. The default `None` is to
        compute the percentile(s) over a flattened version of the array.
    eps : float, default=1e-20
        Small value to avoid division by zero.
    dtype : type, default=np.float32
        Data type of the output.

    Returns
    -------
    NDArray
        Normalized image.
    """
    # Compute percentiles
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)

    return normalize_min_max(x, mi, ma, eps=eps, dtype=dtype)


def normalize_min_max(
    x: NDArray,
    min_val: float,
    max_val: float,
    eps: float = 1e-20,
    dtype: type = np.float32,
) -> NDArray:
    """
    Normalize input image using absolute min and max values.

    This approach was used in the CARE method.

    Parameters
    ----------
    x : NDArray
        Input image.
    min_val : float
        Lower value.
    max_val : float
        Upper value.
    eps : float, default=1e-20
        Small value to avoid division by zero.
    dtype : type, default=np.float32
        Data type of the output.

    Returns
    -------
    NDArray
        Normalized image.
    """
    # change types
    x = x.astype(dtype, copy=False)
    min_val = (
        dtype(min_val) if np.isscalar(min_val) else min_val.astype(dtype, copy=False)
    )
    max_val = (
        dtype(max_val) if np.isscalar(max_val) else max_val.astype(dtype, copy=False)
    )
    eps = dtype(eps)

    return (x - min_val) / (max_val - min_val + eps)


def normalize_min_mse(x: NDArray, target: NDArray) -> NDArray:
    """
    Normalize input image to a target image using minimum mean square error.

    This method performs an affine rescaling of x, such that the mean squared error to
    target is minimal.

    Parameters
    ----------
    x : NDArray
        Input image.
    target : NDArray
        Target image.

    Returns
    -------
    NDArray
        Normalized input image.
    """
    # compute covariance
    cov = np.cov(x.flatten(), target.flatten())

    # scaling factor
    alpha = cov[0, 1] / (cov[0, 0] + 1e-10)

    # offset
    beta = target.mean() - alpha * x.mean()

    return alpha * x + beta
