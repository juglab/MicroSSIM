from typing import Union, Optional

import numpy as np
from numpy.typing import NDArray

from microssim._micro_ssim_internal import get_transformation_params
from microssim.ssim import compute_ssim_elements, compute_ssim, SSIM
from microssim._ri_factor import get_ri_factor


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
    def __init__(
        self,
        bkg_percentile=3,
        offset_pred=None,
        offset_gt=None,
        max_val=None,
        ri_factor=None,
    ) -> None:
        """
        Args:
            bkg_percentile (int, optional): Percentile of the image to be used as background. Defaults to 3.
            offset_pred (float, optional): An estimate of background pixel intensity for prediction. This will be subtracted from the prediction. When None, the bkg_percentile of the prediction will be used. Defaults to None.
            offset_gt (float, optional): An estimate of background pixel intensity for groundtruth. This will be subtracted from the ground truth. When None, the bkg_percentile of the ground truth will be used. Defaults to None.
            max_val (float, optional): Maximum value, used in normalization. Defaults to None.
            ri_factor (float, optional): Factor to be multiplied with the prediction. This is estimated from the data if not provided. Defaults to None.

        # Example:
            ssim = MicroSSIM()
            ssim.fit(gt_all, pred_all)
            print(ssim.score(gt, pred))
        """
        self._bkg_percentile = bkg_percentile
        self._offset_pred = offset_pred
        self._offset_gt = offset_gt
        self._max_val = max_val
        self._ri_factor = ri_factor
        self._initialized = self._ri_factor is not None
        if self._initialized:
            assert (
                self._offset_gt is not None
                and self._offset_pred is not None
                and self._max_val is not None
            ), "If ri_factor is provided, offset_pred, offset_gt and max_val must be provided as well."

    def get_init_params_dict(self):
        """
        Returns the initialization parameters of the measure. This can be used to save the model and
        reload it later or to initialize other SSIM variants with the same parameters.
        """
        assert self._initialized is True, "model is not initialized."
        return {
            "bkg_percentile": self._bkg_percentile,
            "offset_pred": self._offset_pred,
            "offset_gt": self._offset_gt,
            "max_val": self._max_val,
            "ri_factor": self._ri_factor,
        }

    def _set_hparams(self, gt: np.ndarray, pred: np.ndarray):
        if self._offset_gt is None:
            self._offset_gt = np.percentile(gt, self._bkg_percentile, keepdims=False)

        if self._offset_pred is None:
            self._offset_pred = np.percentile(
                pred, self._bkg_percentile, keepdims=False
            )

        if self._max_val is None:
            self._max_val = (gt - self._offset_gt).max()

    def fit(self, gt: np.ndarray, pred: np.ndarray):
        assert self._initialized is False, "fit method can be called only once."

        if isinstance(gt, np.ndarray):
            self._set_hparams(gt, pred)

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
            self._set_hparams(gt_squished, pred_squished)

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
        self._ri_factor = get_transformation_params(gt_norm, pred_norm)

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
