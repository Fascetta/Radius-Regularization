import torch
import torch.nn.functional as F
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
    """Calculate intersection and union for GPU tensors."""
    assert output.dim() in [1, 2, 3], f"Output dim must be 1, 2, or 3 but got {output.dim()}"
    assert output.shape == target.shape, f"Shapes do not match: {output.shape} != {target.shape}"

    output = output.view(-1)
    target = target.view(-1)

    mask = target != ignore_index
    output = output[mask]
    target = target[mask]

    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target


def calculate_ece(logits, labels, n_bins=15):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()
