import torch
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


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(
        intersection.float().cpu(), bins=K, min=0, max=K - 1
    )
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return (
        area_intersection.cpu().numpy(),
        area_union.cpu().numpy(),
        area_target.cpu().numpy(),
    )
