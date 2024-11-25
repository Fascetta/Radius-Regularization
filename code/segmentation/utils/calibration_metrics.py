"""Calibration metrics for classification models."""

from torchmetrics.classification import MulticlassCalibrationError


class CalibrationMetrics:
    """_summary_"""

    def __init__(self, n_classes, n_bins=15):
        self.n_classes = n_classes
        self.mce = MulticlassCalibrationError(
            num_classes=n_classes, norm="max", n_bins=n_bins
        )
        self.ece = MulticlassCalibrationError(
            num_classes=n_classes, norm="l1", n_bins=n_bins
        )
        self.rmsce = MulticlassCalibrationError(
            num_classes=n_classes, norm="l2", n_bins=n_bins
        )

    def update(self, preds, targets):
        """_summary_

        Args:
            logits (_type_): _description_
            targets (_type_): _description_
        """
        self.mce.update(preds, targets)
        self.ece.update(preds, targets)
        self.rmsce.update(preds, targets)

    def compute(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        mce = self.mce.compute()
        ece = self.ece.compute()
        rmsce = self.rmsce.compute()

        return {"mce": mce.item(), "ece": ece.item(), "rmsce": rmsce.item()}
