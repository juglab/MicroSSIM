"""Background calculation and removal functions."""

import numpy as np
from numpy.typing import NDArray


def get_background(x: NDArray, percentage: float = 3) -> float:
    """
    Compute the value of the background based on the percentile.

    This method simply run `np.percentile` on the input image.

    Parameters
    ----------
    x : numpy.ndarray
        The input image.
    percentage : float, default=3
        Percentage for the percentile.

    Returns
    -------
    float
        The value of the background.
    """
    return np.percentile(x, percentage, keepdims=True)


def remove_background(
    x: NDArray, percentage: float = 3, dtype: type = np.float32
) -> NDArray:
    """
    Remove the background from an image.

    Parameters
    ----------
    x : NDArray
        The input image.
    percentage : float, default=3
        Percentage for the percentile.
    dtype : type, default=np.float32
        Data type for the output image.

    Returns
    -------
    NDArray
        The image with the background removed.
    """

    # get percentile
    perc_val = get_background(x, percentage).astype(dtype, copy=False)

    return x.astype(dtype, copy=False) - perc_val
