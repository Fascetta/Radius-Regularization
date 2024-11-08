import torch

def differentiable_ece(predictions, targets, n_bins=15):
    # Soft version of ECE calculation for differentiable usage.
    # predictions: Model output logits
    # targets: True labels
    # n_bins: Number of bins for confidence calibration

    # Apply softmax to get class probabilities
    probs = torch.nn.functional.softmax(predictions, dim=1)

    # Get maximum probability and corresponding class for each prediction
    confidences, predicted_classes = torch.max(probs, 1)
    correct = predicted_classes.eq(targets)

    # Binning confidences
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.tensor(0.0, device=predictions.device)

    for i in range(n_bins):
        # Get the indices for predictions within this bin
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if in_bin.any():
            bin_acc = correct[in_bin].float().mean()
            bin_conf = confidences[in_bin].mean()
            bin_ece = torch.abs(bin_conf - bin_acc) * in_bin.float().mean()
            ece += bin_ece  # Aggregate ECE across bins

    return ece
