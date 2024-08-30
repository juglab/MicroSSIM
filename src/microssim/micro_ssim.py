from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from microssim.ri_factor.ri_factor import get_global_ri_factor, get_ri_factor
from microssim.ssim import SSIM, compute_ssim, compute_ssim_elements

# TODO add convenience function or example to show case the background percentile
# TODO add docstring examples
# TODO function micro_structural_similarity with the bg_factor and the calculation,
# would be more handy


def micro_structural_similarity(
    image1,
    image2,
    *,
    win_size=None,
    data_range=None,
    channel_axis=None,
    gaussian_weights=True,
    ri_factor: Optional[float] = None,
    individual_components: bool = False,
    **kwargs,
) -> Union[NDArray, SSIM]:
    """
    Compute the MicroSSIM metrics.

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
    individual_components : bool, default = False
        If True, return the individual SSIM components.

    Other Parameters
    ----------------
    use_sample_covariance : bool
        If True, normalize covariances by N-1 rather than, N where N is the
        number of pixels within the sliding window.
    K1 : float
        Algorithm parameter, K1 (small constant).
    K2 : float
        Algorithm parameter, K2 (small constant).
    sigma : float
        Standard deviation for the Gaussian when `gaussian_weights` is True.

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

    return compute_ssim(
        elements,
        alpha=ri_factor,
        individual_components=individual_components,
    )


class MicroSSIM:
    """
    A class to perform all preprocessing and MicroSSIM calculations.

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
    """

    def __init__(
        self: Self,
        bg_percentile: int = 3,
        offset_pred: Optional[float] = None,
        offset_gt: Optional[float] = None,
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
        offset_pred : float, optional
            Estimate of background pixel intensity in the prediction image.
        offset_gt : float, optional
            Estimate of background pixel intensity in the ground truth image.
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

    def get_parameters_as_dict(self: Self) -> dict[str, float]:
        """
        Return the attributes of the class as a dictionary.

        Returns
        -------
        dictionary of {str: float}
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

    def _compute_parameters(self, gt: NDArray, pred: NDArray) -> None:
        """
        Compute the MicroSSIM attributes that are missing.

        Parameters
        ----------
        gt : numpy.ndarray
            Reference image array.
        pred : numpy.ndarray
            Image array to compare to the reference.
        """
        if self._offset_gt is None:
            self._offset_gt = np.percentile(gt, self._bg_percentile, keepdims=False)

        if self._offset_pred is None:
            self._offset_pred = np.percentile(pred, self._bg_percentile, keepdims=False)

        if self._max_val is None:
            self._max_val = (gt - self._offset_gt).max()

    def fit(
        self: Self,
        gt: Union[list[NDArray], NDArray],
        pred: Union[list[NDArray], NDArray],
    ) -> None:
        """
        Fit

        Parameters
        ----------
        gt : np.ndarray
            _description_
        pred : np.ndarray
            _description_
        """
        assert self._initialized is False, "fit method can be called only once."

        if isinstance(gt, np.ndarray):
            self._compute_parameters(gt, pred)

        elif isinstance(gt, list):
            gt_squished = np.concatenate(
                [
                    x.reshape(
                        -1,
                    )
                    for x in gt
                ]
            )
            pred_squished = np.concatenate(
                [
                    x.reshape(
                        -1,
                    )
                    for x in pred
                ]
            )
            self._compute_parameters(gt_squished, pred_squished)

        self._fit(gt, pred)
        self._initialized = True

    def normalize_prediction(self, pred: Union[list[np.ndarray], np.ndarray]):
        if isinstance(pred, list):
            assert isinstance(pred[0], np.ndarray), "List must contain numpy arrays."
            return [self.normalize_prediction(x) for x in pred]

        return (pred - self._offset_pred) / self._max_val

    def normalize_gt(self, gt: Union[list[np.ndarray], np.ndarray]):
        if isinstance(gt, list):
            assert isinstance(gt[0], np.ndarray), "List must contain numpy arrays."
            return [self.normalize_gt(x) for x in gt]

        return (gt - self._offset_gt) / self._max_val

    def _fit(self, gt: np.ndarray, pred: np.ndarray):
        gt_norm = self.normalize_gt(gt)
        pred_norm = self.normalize_prediction(pred)
        self._ri_factor = get_global_ri_factor(gt_norm, pred_norm)

    def score(
        self,
        gt: np.ndarray,
        pred: np.ndarray,
        return_individual_components: bool = False,
    ):
        if not self._initialized:
            raise ValueError(
                "fit method was not called before score method. Expected behaviour is to call fit \
                  with ALL DATA and then call score(), with individual images.\
                  Using all data for fitting ensures better estimation of ri_factor."
            )
        assert (
            gt.shape == pred.shape
        ), "Groundtruth and prediction must have same shape."
        assert len(gt.shape) == 2, "Only 2D images are supported."

        gt_norm = self.normalize_gt(gt)
        pred_norm = self.normalize_prediction(pred)
        return micro_structural_similarity(
            gt_norm,
            pred_norm,
            ri_factor=self._ri_factor,
            data_range=gt_norm.max() - gt_norm.min(),
            return_individual_components=return_individual_components,
        )
