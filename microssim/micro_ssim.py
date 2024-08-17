import numpy as np

from microssim._micro_ssim_internal import get_transformation_params, micro_SSIM


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
        self._fit_called = self._ri_factor is not None
        if self._fit_called:
            assert (
                self._offset_gt is not None
                and self._offset_pred is not None
                and self._max_val is not None
            ), "If ri_factor is provided, offset_pred, offset_gt and max_val must be provided as well."

    def fit(self, gt: np.ndarray, pred: np.ndarray):
        assert self._fit_called is False, "fit method can be called only once."
        if self._offset_gt is None:
            self._offset_gt = np.percentile(gt, self._bkg_percentile, keepdims=False)

        if self._offset_pred is None:
            self._offset_pred = np.percentile(
                pred, self._bkg_percentile, keepdims=False
            )

        if self._max_val is None:
            self._max_val = (gt - self._offset_gt).max()

        self._fit(gt, pred)
        self._fit_called = True

    def normalize_prediction(self, pred: np.ndarray):
        return (pred - self._offset_pred) / self._max_val

    def normalize_gt(self, gt: np.ndarray):
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
        if not self._fit_called:
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
        return micro_SSIM(
            gt_norm,
            pred_norm,
            ri_factor=self._ri_factor,
            data_range=gt_norm.max() - gt_norm.min(),
            return_individual_components=return_individual_components,
        )
