import torch
from torch.nn import functional as F


class ECEMeter:
    def __init__(self, n_bins=15):
        """
        Initialize the ECEMeter with a specified number of bins.

        Parameters:
        n_bins (int): Number of bins to use for the ECE calculation. Default is 15.
        """
        self.n_bins = n_bins
        self.reset()

    def reset(self):
        """
        Reset the internal state of ECE meter

        This method initializes the bins for confidence, accuracy, and count to zeros.
        """
        self.confidence_bins = torch.zeros(self.n_bins)
        self.accuracy_bins = torch.zeros(self.n_bins)
        self.count_bins = torch.zeros(self.n_bins)

    def update(self, logits, labels):
        """
        Update the ECE calcultaion with a new set of predictions and labels.

        Parameters:
        logits (Tensor): The logits output from the model (before applying softmax).
        labels (Tensor): The ground truth labels.
        """
        confidences, predictions = torch.max(F.softmax(logits, dim=1), 1)
        accuracies = predictions.eq(labels)

        bins = torch.linspace(0, 1, self.n_bins + 1)
        for i in range(self.n_bins):
            in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
            self.count_bins[i] += in_bin.sum().item()
            self.confidence_bins[i] += confidences[in_bin].sum().item()
            self.accuracy_bins[i] += accuracies[in_bin].sum().item()

    def compute(self):
        """
        Compute the Expected Calibration Error (ECE).

        """
        ece = 0.0
        for i in range(self.n_bins):
            if self.count_bins[i] > 0:
                avg_confidence = self.confidence_bins[i] / self.count_bins[i]
                avg_accuracy = self.accuracy_bins[i] / self.count_bins[i]
                ece += torch.abs(avg_confidence - avg_accuracy) * (
                    self.count_bins[i] / self.count_bins.sum()
                )
        return ece.item()
