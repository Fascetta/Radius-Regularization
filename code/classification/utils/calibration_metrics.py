"""Calibration metrics for classification models."""

from torchmetrics.classification import MulticlassCalibrationError
from tqdm import tqdm


class CalibrationMetrics:
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

    def update(self, logits, targets):
        self.mce.update(logits, targets)
        self.ece.update(logits, targets)
        self.rmsce.update(logits, targets)

    def compute(self):
        mce = self.mce.compute()
        ece = self.ece.compute()
        rmsce = self.rmsce.compute()

        return {"mce": mce.item(), "ece": ece.item(), "rmsce": rmsce.item()}