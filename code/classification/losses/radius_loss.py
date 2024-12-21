import torch
from torch.nn import functional as F
from torchmetrics.functional.classification.calibration_error import _ce_compute


class RadiusAccuracyLoss:
    """Loss function for radius-based accuracy calibration."""

    @staticmethod
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

    def __call__(self, logits, labels, radii):
        accuracy = self.compute_batch_accuracy_per_class(logits, labels)
        true_class_accuracy = accuracy[labels]
        loss = F.mse_loss(radii, true_class_accuracy)
        return loss


class WeightedRadiusAccuracyLoss(RadiusAccuracyLoss):
    """Loss function for weighted radius-based accuracy calibration."""

    @staticmethod
    def compute_weighted_radius_per_class(logits, labels, radii):
        """
        Computes the weighted average radius for each class in the batch.

        Args:
            logits: Logits from the model (batch_size, n_classes).
            labels: True class labels (batch_size,).
            radii: Radii of the embeddings in hyperbolic space (batch_size,).

        Returns:
            weighted_radius: Weighted average radius for each class (n_classes,).
        """
        n_classes = logits.shape[-1]
        weights = F.softmax(logits, dim=-1)  # Confidence scores
        sample_weights = weights[
            torch.arange(labels.size(0)), labels
        ]  # Weight per sample

        # Initialize accumulators
        total_weights = torch.zeros(n_classes, device=logits.device)
        weighted_radii = torch.zeros(n_classes, device=logits.device)

        # Accumulate weighted radii and weights
        total_weights.index_add_(0, labels, sample_weights)
        weighted_radii.index_add_(0, labels, sample_weights * radii)

        # Compute weighted radius per class
        weighted_radius = torch.zeros(n_classes, device=logits.device)
        weighted_radius[total_weights > 0] = (
            weighted_radii[total_weights > 0] / total_weights[total_weights > 0]
        )

        return weighted_radius

    def __call__(self, logits, labels, radii):
        """
        Computes the RadiusAccuracyLoss with weighted radii.

        Args:
            logits: Logits from the model (batch_size, n_classes).
            labels: True class labels (batch_size,).
            radii: Radii of the embeddings in hyperbolic space (batch_size,).

        Returns:
            Loss: Mean squared error between weighted radii and accuracy per class.
        """
        # Compute batch accuracy per class (inherited from superclass)
        accuracy = self.compute_batch_accuracy_per_class(logits, labels)

        # Compute weighted radii per class
        weighted_radius = self.compute_weighted_radius_per_class(logits, labels, radii)

        # Calculate MSE between weighted radii and accuracy per class
        loss = F.mse_loss(
            weighted_radius, accuracy, reduction="sum"
        )  # Reduce by sum for stability
        return loss


class RadiusConfidenceLoss:
    """Loss function for radius-based confidence calibration."""

    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.radii_running_max = 0.0

    def __call__(self, logits, labels, radii):
        # confidences = torch.max(logits.softmax(dim=-1), dim=-1).values
        confidences = logits.softmax(dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
        self.radii_running_max = max(self.radii_running_max, radii.max().item())
        radii_rescaled = radii / self.radii_running_max
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)
        ce = _ce_compute(confidences, radii_rescaled, bin_boundaries, norm="l2")
        return ce
