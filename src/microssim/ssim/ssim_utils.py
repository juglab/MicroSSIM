"""Calculate the elements used for computing the SSIM.

Code adapted from `skimage.metrics.structural_similarity` under BSD-3-Clause license.
See https://github.com/scikit-image/scikit-image.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter
from skimage._shared.filters import gaussian
from skimage._shared.utils import _supported_float_type, check_shape_equality, warn
from skimage.util.dtype import dtype_range
from typing_extensions import Self


@dataclass
class SSIMElements:
    """Dataclass holding the various values necessary for the computation of the SSIM."""

    ux: NDArray
    """Weighted mean of the first image."""

    uy: NDArray
    """Weighted mean of the second image."""

    vxy: NDArray
    """Weighted covariance of the two images."""

    vx: NDArray
    """Weighted variance of the first image."""

    vy: NDArray
    """Weighted variance of the second image."""

    C1: float
    """Algorithm parameter, C1."""

    C2: float
    """Algorithm parameter, C2."""

    def get_args_tuple(self: Self) -> tuple[Union[NDArray, float], ...]:
        """Return the elements as a list.

        Note that this excludes C3, which is usually unused.

        Returns
        -------
        list of numpy.ndarray or float
            List of the elements.
        """
        return [self.ux, self.uy, self.vxy, self.vx, self.vy, self.C1, self.C2]


@dataclass
class ScaledSSIM:
    """Scaled SSIM metrics result and its components."""

    SSIM: NDArray
    """The SSIM value."""

    luminance: NDArray
    """The luminance value."""

    contrast: NDArray
    """The contrast value."""

    structure: NDArray
    """The structure value."""

    alpha: float
    """MicroSSIM scaling parameter."""

    elements: SSIMElements
    """The elements used for computing the SSIM."""


# TODO since the function that computes SSIM does not use the various SSIM parameters
# (channel, gaussian weights, win_size, etc.) they should probably be removed from
# here
def compute_ssim_elements(
    image1: NDArray,
    image2: NDArray,
    *,
    win_size: Optional[int] = None,
    data_range: Optional[float] = None,
    channel_axis: Optional[int] = None,
    gaussian_weights: bool = False,
    **kwargs: dict[str, float],
) -> SSIMElements:
    """
    Calculate the elements used for computing the SSIM.

    Code adapted from `skimage.metrics.structural_similarity` under BSD-3-Clause
    license.

    Parameters
    ----------
    image1 : numpy.ndarray
        First image being compared.
    image2 : numpy.ndarray
        Second image being compared.
    win_size : int or None, optional
        The side-length of the sliding window used in comparison. Must be an
        odd value. If `gaussian_weights` is True, this is ignored and the
        window size will depend on `sigma`.
    data_range : float, optional
        The data range of the input image (difference between maximum and
        minimum possible values). By default, this is estimated from the image
        data type. This estimate may be wrong for floating-point image data.
        Therefore it is recommended to always pass this scalar value explicitly
        (see note below).
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a
        normalized Gaussian kernel of width sigma=1.5.
    **kwargs : dict
        Additional keyword arguments passed to the SSIM computation.

    Returns
    -------
    SSIMElements
        The elements used for computing the SSIM (means, variances, covariaces, etc.).
    """
    check_shape_equality(image1, image2)
    float_type = _supported_float_type(image1.dtype)

    if channel_axis is not None:
        raise NotImplementedError(
            "Multichannel images are not supported at this time. "
            "Please set `channel_axis` to None."
        )

    K1 = kwargs.pop("K1", 0.01)  # TODO inherited from skimage, but might make explicit
    K2 = kwargs.pop("K2", 0.03)
    K3 = kwargs.pop("K3", None)

    sigma = kwargs.pop("sigma", 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop("use_sample_covariance", True)

    if gaussian_weights:
        # Set to give an 11-tap filter with the default sigma of 1.5 to match
        # Wang et. al. 2004.
        truncate = 3.5

    if win_size is None:
        if gaussian_weights:
            # set win_size used by crop to match the filter size
            r = int(truncate * sigma + 0.5)  # radius as in ndimage
            win_size = 2 * r + 1
        else:
            win_size = 7  # backwards compatibility

    if np.any((np.asarray(image1.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent. "
            "Either ensure that your images are "
            "at least 7x7; or pass win_size explicitly "
            "in the function call, with an odd value "
            "less than or equal to the smaller side of your "
            "images. If your images are multichannel "
            "(with color channels), set channel_axis to "
            "the axis number corresponding to the channels."
        )

    if not (win_size % 2 == 1):
        raise ValueError("Window size must be odd.")

    if data_range is None:
        if np.issubdtype(image1.dtype, np.floating) or np.issubdtype(
            image2.dtype, np.floating
        ):
            raise ValueError(
                "Since image dtype is floating point, you must specify "
                "the data_range parameter. Please read the documentation "
                "carefully (including the note). It is recommended that "
                "you always specify the data_range anyway."
            )
        if image1.dtype != image2.dtype:
            warn(
                "Inputs have mismatched dtypes. Setting data_range "
                "based on img_x.dtype.",
                stacklevel=2,
            )
        dmin, dmax = dtype_range[image1.dtype.type]
        data_range = dmax - dmin
        if np.issubdtype(image1.dtype, np.integer) and (image1.dtype != np.uint8):
            warn(
                "Setting data_range based on img_x.dtype. "
                + f"data_range = {data_range:.0f}. "
                + "Please specify data_range explicitly to avoid mistakes.",
                stacklevel=2,
            )

    ndim = image1.ndim

    if gaussian_weights:
        filter_func = gaussian
        filter_args = {"sigma": sigma, "truncate": truncate, "mode": "reflect"}
    else:
        filter_func = uniform_filter
        filter_args = {"size": win_size}

    # ndimage filters need floating point data
    image1 = image1.astype(float_type, copy=False)
    image2 = image2.astype(float_type, copy=False)

    NP = win_size**ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(image1, **filter_args)
    uy = filter_func(image2, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(image1 * image1, **filter_args)
    uyy = filter_func(image2 * image2, **filter_args)
    uxy = filter_func(image1 * image2, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    # optional prameter set to C2/2 in scikit-image
    C3 = None if K3 is None else (K3 * R) ** 2

    pad = (win_size - 1) // 2
    ux = ux[pad:-pad, pad:-pad].copy()
    uy = uy[pad:-pad, pad:-pad].copy()
    vxy = vxy[pad:-pad, pad:-pad].copy()
    vx = vx[pad:-pad, pad:-pad].copy()
    vy = vy[pad:-pad, pad:-pad].copy()

    return SSIMElements(
        ux=ux,
        uy=uy,
        vxy=vxy,
        vx=vx,
        vy=vy,
        C1=C1,
        C2=C2,
    )


def _scaled_ssim(
    alpha: float,
    elements: SSIMElements,
) -> ScaledSSIM:
    """Compute SSIM from its elements as done in scikit-image.

    Code adapted from `skimage.metrics.structural_similarity` under BSD-3-Clause
    license.

    Parameters
    ----------
    alpha : float
        MicroSSIM scaling parameter.
    elements : SSIMElements
        Elements used for computing the SSIM (means, stds, cov etc.).

    Returns
    -------
    SSIM
        SSIM object.
    """
    # compute SSIM as in scikit-image, albeit with the scaling factor
    A1, A2, B1, B2 = (
        2 * alpha * elements.ux * elements.uy + elements.C1,
        2 * alpha * elements.vxy + elements.C2,
        elements.ux**2 + (alpha**2) * elements.uy**2 + elements.C1,
        elements.vx + (alpha**2) * elements.vy + elements.C2,
    )

    D = B1 * B2
    S = (A1 * A2) / D

    # compute physical terms
    term = 2 * alpha * np.sqrt(elements.vx * elements.vy) + elements.C2
    luminance = A1 / B1
    contrast = term / B2
    structure = A2 / term

    return ScaledSSIM(
        SSIM=S,
        luminance=luminance,
        contrast=contrast,
        structure=structure,
        alpha=alpha,
        elements=elements,
    )


def _ssim_with_c3(
    alpha: float,
    elements: SSIMElements,
    C3: float,
) -> ScaledSSIM:
    """Compute SSIM with C3.

    C3 is comonly set to C2/2 (e.g. scikit-image and torch implementations). This
    function allows setting a different value for C3.

    C3 could be computed as `C3 = None if K3 is None else (K3 * R) ** 2`.

    Parameters
    ----------
    alpha : float
        MicroSSIM scaling parameter.
    elements : SSIMElements
        Elements used for computing the SSIM (means, stds, covars etc.).
    C3 : float
        C3 parameter.

    Returns
    -------
    SSIM
        SSIM object.
    """
    lum_num = 2 * alpha * elements.ux * elements.uy + elements.C1
    lum_denom = elements.ux**2 + (alpha**2) * elements.uy**2 + elements.C1

    contrast_num = 2 * alpha * np.sqrt(elements.vx * elements.vy) + elements.C2
    contrast_denom = elements.vx + (alpha**2) * elements.vy + elements.C2

    structure_denom = alpha * np.sqrt(elements.vx * elements.vy) + C3
    structure_num = alpha * elements.vxy + C3

    num = lum_num * contrast_num * structure_num
    denom = lum_denom * contrast_denom * structure_denom
    S = num / denom

    return ScaledSSIM(
        SSIM=S,
        luminance=lum_num / lum_denom,
        contrast=contrast_num / contrast_denom,
        structure=structure_num / structure_denom,
        alpha=alpha,
        elements=elements,
    )


def compute_scaled_ssim(
    elements: SSIMElements,
    *,
    alpha: float = 1.0,
    return_individual_components: bool = False,
) -> Union[NDArray, ScaledSSIM]:
    """Compute scaled SSIM from SSIM elements and a scaling factor.

    The SSIM elements are calculated using `compute_ssim_elements`.

    Parameters
    ----------
    elements : SSIMElements
        Elements used for computing the SSIM (means, stds, covars etc.).
    alpha : float
        MicroSSIM scaling parameter.
    return_individual_components : bool, default = False
        If True, return the individual SSIM components.

    Returns
    -------
    numpy.ndarray or SSIM
        SSIM value.
    """
    # TODO: scikit implements additional parameters (win_size, gaussian_weights, etc.)
    # that change the value of the elements. This is not implemented here, but we need
    # to consider their implementation to be closer to the SSIM.
    # see skimage.metrics.structural_similarity for more details

    # compute SSIM
    ssim = _scaled_ssim(alpha=alpha, elements=elements)
    mean_ssim = ssim.SSIM.mean(dtype=np.float64)

    if return_individual_components:
        return ssim
    else:
        return mean_ssim
