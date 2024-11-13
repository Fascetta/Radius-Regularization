import torch


class RadiusLabelSmoothing:
    """Class which computes the smoothed labels for radius-based calibration."""

    def __init__(self, device, n_classes=10, ema_alpha=0.1):
        # EMA of radii per class
        self.device = device
        self.radii_ema = torch.zeros(n_classes, device=self.device)
        self.alpha = ema_alpha
        self.n_classes = n_classes

    @torch.no_grad()
    def update_ema(self, radii):
        self.radii_ema = self.alpha * radii + (1 - self.alpha) * self.radii_ema

    def compute_mean_radius_per_class(self, radii, labels):
        mean_radius = torch.zeros(self.n_classes, device=self.device)
        for i in range(self.n_classes):
            class_mask = labels == i
            mean_radius[i] = radii[class_mask].mean()
        return mean_radius

    def __call__(self, labels, radii):
        radii_per_class = self.compute_mean_radius_per_class(radii, labels)
        self.update_ema(radii_per_class)

        # for each label, create a tensor with size equal to the number of classes
        # the correct label will have the smoothed radius, while the rest will have
        # (1 - smoothed radius) / (n_classes - 1)

        smoothed_labels = torch.zeros(labels.size(0), self.n_classes, device=self.device)
        for i in range(smoothed_labels.size(0)):
            radii_true_class = self.radii_ema[labels[i]]
            radii_other_classes = (1 - radii_true_class) / (self.n_classes - 1)
            smoothed_labels[i] = radii_other_classes
            smoothed_labels[i, labels[i]] = radii_true_class
        return smoothed_labels

