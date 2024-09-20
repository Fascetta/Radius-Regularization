import torch
from torch.nn import functional as F
from torchmetrics.functional.classification.calibration_error import _ce_compute


def compute_batch_accuracy_per_class(logits, labels):
    """Computes the estimated confidence for each class."""
    n_classes = logits.shape[-1]
    accuracy = torch.zeros(n_classes, device=logits.device)
    preds = logits.argmax(dim=-1)

    # compute total counts and correct counts for each class
    total_counts = torch.bincount(labels, minlength=n_classes).float()
    correct = (preds == labels).float()
    correct_counts = torch.zeros(n_classes, device=logits.device)
    correct_counts.index_add_(0, labels, correct.float())

    # compute confidence for each class
    accuracy = correct_counts / total_counts
    accuracy[total_counts == 0] = 0.0
    return accuracy


def radius_accuracy_loss(logits, labels, radii):
    accuracy = compute_batch_accuracy_per_class(logits, labels)
    true_class_accuracy = accuracy[labels]
    loss = F.mse_loss(radii, true_class_accuracy)
    return loss


class RadiusConfidenceLoss:
    """Loss function for radius-based confidence calibration."""

    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.radii_running_max = 0.

    def __call__(self, logits, radii, labels):
        # confidences = torch.max(logits.softmax(dim=-1), dim=-1).values
        confidences = logits.softmax(dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
        self.radii_running_max = max(self.radii_running_max, radii.max().item())
        radii_rescaled = radii / self.radii_running_max
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)
        ce = _ce_compute(confidences, radii_rescaled, bin_boundaries, norm="l2")
        return ce
