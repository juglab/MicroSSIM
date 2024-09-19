"""MicroSSIM function and class."""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from microssim.image_processing import (
    compute_norm_parameters,
    linearize_list,
    normalize_min_max,
)
from microssim.ri_factor.ri_factor import get_global_ri_factor, get_ri_factor
from microssim.ssim import ScaledSSIM, compute_scaled_ssim, compute_ssim_elements

# TODO add convenience function or example to show case the background percentile
# TODO add docstring examples
# TODO 3D dims will have issue with skimage ssim if win_size is not provided


# TODO the various parameters inherited from skimage are not used in the reconstruction
# of SSIM and should probably be removed
def _compute_micro_ssim(
    image1: NDArray,
    image2: NDArray,
    *,
    win_size: Optional[int] = None,
    data_range: Optional[float] = None,
    channel_axis: Optional[int] = None,
    gaussian_weights: bool = True,  # TODO why is it true by default?
    ri_factor: Optional[float] = None,
    return_individual_components: bool = False,
    **kwargs: dict,
) -> Union[NDArray, ScaledSSIM]:
    """
    Compute the MicroSSIM metrics.

    This methods expects background-subtracted images.

    If the range-invariant factor is not provided, it will be estimated from the images.

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
    ri_factor : float or None, optional
        MicroSSIM scaling factor. If None, it will be estimated from the images.
    return_individual_components : bool, default = False
        If True, return the individual SSIM components.
    **kwargs : dict
        Additional keyword arguments passed to the SSIM computation, following skimage
        SSIM function's signature.

    Returns
    -------
    numpy.ndarray or SSIM
        SSIM value.
    """
    elements = compute_ssim_elements(
        image1,
        image2,
        win_size=win_size,
        data_range=data_range,
        channel_axis=channel_axis,
        gaussian_weights=gaussian_weights,
        **kwargs,
    )

    if ri_factor is None:
        ri_factor = get_ri_factor(elements)

    return compute_scaled_ssim(
        elements,
        alpha=ri_factor,
        return_individual_components=return_individual_components,
    )


def micro_structural_similarity(
    gt: Union[NDArray, list[NDArray]],
    pred: Union[NDArray, list[NDArray]],
    *,
    bg_percentile: int = 3,
    offset_gt: Optional[float] = None,
    offset_pred: Optional[float] = None,
    max_val: Optional[float] = None,
    ri_factor: Optional[float] = None,
    return_individual_components: bool = False,
) -> Union[float, ScaledSSIM, list[float], list[ScaledSSIM]]:
    """
    Compute the mean MicroSSIM metric between two images.

    MicroSSIM computes a scaled version of the structural similarity index (SSIM), in
    which images are first normalized using an offset and maximum value. A
    range-invariant factor is then estimated by maximizing a scaled SSIM metrics
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
    return_individual_components : bool, default=False
        If True, return the individual SSIM components.

    Returns
    -------
    float or list of float or ScaledSSIM or list of ScaledSSIM
        Mean MicroSSIM metric between the images, either as a list if the input are
        lists or array with more than 2 dimensions. The return type can be either
        float or ScaledSSIM depending on the `return_individual_components` parameter.

    Examples
    --------
    >>> import numpy as np
    >>> from microssim import micro_structural_similarity
    >>> rng = np.random.default_rng(42)
    >>> gt = 150 + rng.integers(0, data_range, (100, 100))
    >>> pred = rng.poisson(gt) / 10. - 100
    >>> micro_structural_similarity(gt, pred)
    """
    # generate parameters for the metrics computation
    micro_ssim = MicroSSIM(
        bg_percentile=bg_percentile,
        offset_gt=offset_gt,
        offset_pred=offset_pred,
        max_val=max_val,
        ri_factor=ri_factor,
    )
    micro_ssim.fit(gt, pred)

    # compute the MicroSSIM metric
    if isinstance(gt, list) or gt.ndim > 2:
        return [
            micro_ssim.score(
                gt_i,
                pred_i,
                return_individual_components=return_individual_components,
            )
            for gt_i, pred_i in zip(gt, pred)
        ]
    else:
        return micro_ssim.score(
            gt,
            pred,
            return_individual_components=return_individual_components,
        )


# TODO why not accept lists in score?
class MicroSSIM:
    """
    A class computing the MicroSSIM metric between images.

    In addition to computing the metrics, this class allows to inspect the parameters
    estimated along the way, such as the offsets, the max value and the range-invariant
    factor.

    Parameters
    ----------
    bg_percentile : int, default=3
        Percentile of the image considered as background.
    offset_gt : float, optional
        Estimate of background pixel intensity in the ground truth image.
    offset_pred : float, optional
        Estimate of background pixel intensity in the prediction image.
    max_val : float, optional
        Maximum value used in normalization.
    ri_factor : float, optional
        MicroSSIM scaling factor.

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

    def __init__(
        self: Self,
        bg_percentile: int = 3,
        offset_gt: Optional[float] = None,
        offset_pred: Optional[float] = None,
        max_val: Optional[float] = None,
        ri_factor: Optional[float] = None,
    ) -> None:
        """
        Constructor.

        If no offsets are provided, the `bg_percentile` of the image will be used as the
        to estimate background.

        If `ri_factor` is provided, the other parameters (except the background
        percentile) must be provided as well.

        Parameters
        ----------
        bg_percentile : int, default=3
            Percentile of the image considered as background.
        offset_gt : float, optional
            Estimate of background pixel intensity in the ground truth image.
        offset_pred : float, optional
            Estimate of background pixel intensity in the prediction image.
        max_val : float, optional
            Maximum value used in normalization.
        ri_factor : float, optional
            MicroSSIM scaling factor.

        Raises
        ------
        ValueError
            If `ri_factor` is provided, but not the other parameters.
        """
        self._bg_percentile = bg_percentile
        self._offset_pred = offset_pred
        self._offset_gt = offset_gt
        self._max_val = max_val
        self._ri_factor = ri_factor
        self._initialized = self._ri_factor is not None

        if self._initialized:
            if (
                self._offset_gt is None
                or self._offset_pred is None
                or self._max_val is None
            ):
                raise ValueError(
                    "Please specify offset_pred, offset_gt and max_val if ri_factor is "
                    "provided."
                )

    def get_parameters(self: Self) -> dict[str, float]:
        """
        Return the attributes of the class as a dictionary.

        Returns
        -------
        {str: float}
            Dictionary containing the attributes of the class.
        """
        if not self._initialized:
            raise ValueError(
                "MicroSSIM has not been initialized, please call the `fit` method "
                "first."
            )

        return {
            "bg_percentile": self._bg_percentile,
            "offset_pred": self._offset_pred,
            "offset_gt": self._offset_gt,
            "max_val": self._max_val,
            "ri_factor": self._ri_factor,
        }

    def fit(
        self: Self,
        gt: Union[list[NDArray], NDArray],
        pred: Union[list[NDArray], NDArray],
    ) -> Self:
        """
        Fit parameters to the images.

        Parameters
        ----------
        gt : numpy.ndarray or list of numpy.ndarray
            Reference arrays.
        pred : numpy.ndarray or list of numpy.ndarray
            Arrays to be comapred to the reference arrays.

        Returns
        -------
        MicroSSIM
            The instance of the class in order to allow chaining.

        Raises
        ------
        ValueError
            If the images are of different types (list or numpy.ndarray).
        ValueError
            If the lists of images have different lengths.
        ValueError
            If the images are arrays with different shapes.
        ValueError
            If the images are not 2D or 3D.
        """
        if type(gt) != type(pred):
            raise ValueError("Images must be of the same type (list or numpy.ndarray).")

        if isinstance(gt, list):
            if len(gt) != len(pred):
                raise ValueError("Lists must have the same length.")

            # linearize and concatenate the list of images
            gt_proc = linearize_list(gt)
            pred_proc = linearize_list(pred)
        else:
            if gt.shape != pred.shape:
                raise ValueError(
                    f"Images must have the same shape (got {gt.shape} and {pred.shape})."
                )

            if gt.ndim < 2 or gt.ndim > 3:
                raise ValueError("Only 2D or 3D images are supported.")

            gt_proc = gt
            pred_proc = pred

        # compute the offsets and maximum value
        self._offset_gt, self._offset_pred, self._max_val = compute_norm_parameters(
            gt_proc,
            pred_proc,
            self._bg_percentile,
            self._offset_gt,
            self._offset_pred,
            self._max_val,
        )
        # normalize the images
        gt_norm = normalize_min_max(gt, self._offset_gt, self._max_val)
        pred_norm = normalize_min_max(pred, self._offset_pred, self._max_val)

        # compute range-invariant factor
        self._ri_factor = get_global_ri_factor(gt_norm, pred_norm)

        self._initialized = True

        return self

    def score(
        self,
        gt: NDArray,
        pred: NDArray,
        return_individual_components: bool = False,
        **kwargs: dict,
    ) -> Union[float, ScaledSSIM]:
        """Compute the metrics between two arrays.

        Only 2D arrays are supported.

        Parameters
        ----------
        gt : numpy.ndarray
            Reference array.
        pred : NDArray
            Array to be compared to the reference array.
        return_individual_components : bool, default=False
            If True, return the individual SSIM components.
        **kwargs : dict
            Additional keyword arguments passed to the SSIM computation, following
            skimage SSIM function's signature.

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

        gt_norm = normalize_min_max(gt, self._offset_gt, self._max_val)
        pred_norm = normalize_min_max(pred, self._offset_pred, self._max_val)

        return _compute_micro_ssim(
            gt_norm,
            pred_norm,
            ri_factor=self._ri_factor,
            data_range=gt_norm.max() - gt_norm.min(),
            return_individual_components=return_individual_components,
            **kwargs,
        )
