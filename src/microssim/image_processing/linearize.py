"""Linearize list of images."""

import numpy as np
from numpy.typing import NDArray


def linearize_list(images: list[NDArray]) -> NDArray:
    """
    Linearize and concatenate a list of images, avoiding copying the arrays.

    Parameters
    ----------
    images : list of numpy.ndarray
        Input images.

    Returns
    -------
    np.ndarray
        Lineatized and concatenated images.
    """
    return np.concatenate([np.ravel(x) for x in images])
