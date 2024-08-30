"""Range invariant factor computation, corresponding to the alpha paramter of
MicroSSIM."""
import numpy as np
from scipy.optimize import minimize

from .ssim.ssim_utils import SSIMElements, _ssim


def get_ri_factor(elements: SSIMElements) -> float:
    """Compute range invariant factor.

    The range invariant factor is the alpha value used in MicroSSIM.

    Parameters
    ----------
    elements : SSIMElements
        SSIM elements.

    Returns
    -------
    float
        Range invariant factor.
    """
    initial_guess = np.array([1])
    res = minimize(
        # _ssim(*args) returns an SSIM class, whose SSIM attribute is a numpy array
        lambda *args: -1 * _ssim(*args).SSIM.mean(), initial_guess, args=elements
    )
    return res.x[0]