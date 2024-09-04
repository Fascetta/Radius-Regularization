import torch
from torch.nn import functional as F


def compute_estimated_confidence(logits, labels):
    """Computes the estimated confidence for each class."""
    n_classes = logits.shape[-1]
    confidence = torch.zeros(n_classes, device=logits.device)
    preds = logits.argmax(dim=-1)

    # compute total counts and correct counts for each class
    total_counts = torch.bincount(labels, minlength=n_classes).float()
    correct = (preds == labels).float()
    correct_counts = torch.zeros(n_classes, device=logits.device)
    correct_counts.index_add_(0, labels, correct.float())

    # compute confidence for each class
    confidence = correct_counts / total_counts
    confidence[total_counts == 0] = 0.0
    return confidence


def radius_confidence_loss(logits, labels, radii):
    confidence = compute_estimated_confidence(logits, labels)
    true_class_confidence = confidence[labels]
    # radii = radii / radii.max()
    loss = F.mse_loss(radii, true_class_confidence)
    return loss
