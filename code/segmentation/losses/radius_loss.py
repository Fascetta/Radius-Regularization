import torch
from torch.nn import functional as F
from torchmetrics.functional.classification.calibration_error import _ce_compute


class RadiusAccuracyLoss:
    """Loss function for radius-based accuracy calibration."""

    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index

    @staticmethod
    def compute_batch_accuracy_per_class(logits, labels, ignore_index=255):
        """Computes the estimated confidence for each class."""
        n_classes = logits.shape[1]
        preds = logits.argmax(dim=1)

        # Flatten predictions and labels
        preds = preds.view(-1)
        labels = labels.view(-1)

        # Create mask for valid labels
        mask = labels != ignore_index
        valid_labels = labels[mask]
        valid_preds = preds[mask]

        # Compute total counts per class
        num_samples_per_class = torch.bincount(valid_labels, minlength=n_classes).float()

        # Compute correct predictions per class
        correct = (valid_preds == valid_labels).float()
        correct_counts = torch.zeros(n_classes, device=logits.device)
        correct_counts.index_add_(0, valid_labels, correct)

        # Compute accuracy per class with safe division
        accuracy = torch.where(
            num_samples_per_class > 0,
            correct_counts / num_samples_per_class,
            torch.zeros_like(num_samples_per_class)  # Handle zero-sample classes
        )

        return accuracy

    def __call__(self, logits, labels, radii):
        # Compute per-class accuracy
        accuracy = self.compute_batch_accuracy_per_class(logits, labels, self.ignore_index)

        # Gather true class accuracy for each pixel
        mask = labels != self.ignore_index
        true_class_accuracy = torch.zeros_like(labels, dtype=torch.float, device=logits.device)
        true_class_accuracy[mask] = accuracy[labels[mask]]

        # Compute the MSE loss between predicted radii and true accuracy
        loss = F.mse_loss(radii, true_class_accuracy)
        return loss


class RadiusConfidenceLoss:
    """Loss function for radius-based confidence calibration."""

    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.radii_running_max = 0.

    def __call__(self, logits, labels, radii):
        # confidences = torch.max(logits.softmax(dim=-1), dim=-1).values
        confidences = logits.softmax(dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
        self.radii_running_max = max(self.radii_running_max, radii.max().item())
        radii_rescaled = radii / self.radii_running_max
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)
        ce = _ce_compute(confidences, radii_rescaled, bin_boundaries, norm="l2")
        return ce
