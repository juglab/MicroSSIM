"""Range invariant factor computation."""

from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from tqdm import tqdm

from microssim.ssim.ssim_utils import SSIMElements, _scaled_ssim, compute_ssim_elements


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
        lambda *args: -1 * _scaled_ssim(*args).SSIM.mean(),
        initial_guess,
        args=elements,
    )
    return res.x[0]


def _aggregate_ssim_elements(
    gt: NDArray,
    pred: NDArray,
    **ssim_kwargs,
) -> SSIMElements:
    """
    Aggregate the SSIM elements of a list of images.

    Parameters
    ----------
    gt : numpy.ndarray
        Reference array.
    pred : numpy.ndarray
        Array to be compared to the reference.
    **ssim_kwargs : Any
        Additional arguments to pass to the SSIM computation.

    Returns
    -------
    SSIMElements
        Aggregated SSIM elements.

    Raises
    ------
    ValueError
        If the ground-truth and prediction arrays are not lists or arrays of more than
        2 dimensions.
    """
    # make sure that we can iterate over the images
    if not isinstance(gt, list) and gt.ndim <= 2:
        raise ValueError(
            "The ground-truth and prediction must either be list of arrays, or arrays "
            "with more than 2 dimensions."
        )

    # remove data range from the kwargs
    if "data_range" in ssim_kwargs:
        ssim_kwargs.pop("data_range")

    ux_arr = []
    uy_arr = []
    vx_arr = []
    vy_arr = []
    vxy_arr = []

    # loop over the array
    for idx in tqdm(range(len(gt))):
        gt_i: NDArray = gt[idx]
        pred_i: NDArray = pred[idx]

        # compute SSIM elements
        elements = compute_ssim_elements(
            gt_i,
            pred_i,
            data_range=gt_i.max() - gt_i.min(),
            **ssim_kwargs,
        )

        # linearize the elements
        ux_arr.append(
            elements.ux.reshape(
                -1,
            )
        )
        uy_arr.append(
            elements.uy.reshape(
                -1,
            )
        )
        vx_arr.append(
            elements.vx.reshape(
                -1,
            )
        )
        vy_arr.append(
            elements.vy.reshape(
                -1,
            )
        )
        vxy_arr.append(
            elements.vxy.reshape(
                -1,
            )
        )

    # concatenate the elements
    global_elements = SSIMElements(
        ux=np.concatenate(ux_arr),
        uy=np.concatenate(uy_arr),
        vx=np.concatenate(vx_arr),
        vy=np.concatenate(vy_arr),
        vxy=np.concatenate(vxy_arr),
        C1=elements.C1,
        C2=elements.C2,
    )

    return global_elements


def get_global_ri_factor(
    gt: Union[list[NDArray], NDArray],
    pred: Union[list[NDArray], NDArray],
    **ssim_kwargs,
) -> float:
    """
    Compute a global range invariant factor.

    The inputs can either be arrays or list of arrays.

    Parameters
    ----------
    gt : numpy.ndarray or list of numpy.ndarray
        Reference array.
    pred : numpy.ndarray or list of numpy.ndarray
        Array to be compared to the reference.
    **ssim_kwargs : Any
        Additional arguments to pass to the SSIM computation.

    Returns
    -------
    float
        Global range invariant factor.

    Raises
    ------
    ValueError
        If the ground-truth and prediction arrays have different types, lengths or
        shapes.
    """
    if type(gt) != type(pred):
        raise ValueError(
            f"Ground-truth and prediction arrays must have the same type "
            f"(got {type(gt)} and {type(pred)}, respectively)."
        )

    if isinstance(gt, list):
        if len(gt) != len(pred):
            raise ValueError(
                f"Ground-truth and prediction lists must have the same length "
                f"(got {len(gt)} and {len(pred)}, respectively)."
            )
    else:
        if gt.shape != pred.shape:
            raise ValueError(
                f"Ground-truth and prediction arrays must have the same "
                f"shape (got {gt.shape} and {pred.shape}, respectively)."
            )

    # TODO should this be refactored?
    if isinstance(gt, list) or gt.ndim > 2:
        # aggregate the SSIM elements
        elements = _aggregate_ssim_elements(gt, pred, **ssim_kwargs)
    else:
        # simply calculate them from the arrays
        elements = compute_ssim_elements(
            gt,
            pred,
            data_range=gt.max() - gt.min(),
            **ssim_kwargs,
        )

    # return the global range invariant factor
    return get_ri_factor(elements)
