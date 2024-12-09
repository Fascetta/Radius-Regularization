import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lib.geoopt.optim import RiemannianAdam, RiemannianSGD
from losses.focal_loss import FocalLoss
from losses.radius_loss import RadiusAccuracyLoss
from utils.initialize import select_dataset, select_model
from utils.metrics import calculate_ece, intersectionAndUnionGPU
from utils.posthoc_calibration_utils import get_optimal_confidence_tau


class TrainLearner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        _, _, _, img_dim, num_classes = select_dataset(
            cfg, validation_split=cfg.validation_split
        )
        self.num_classes = num_classes
        self.model = select_model(img_dim, num_classes, cfg)

        if cfg.base_loss == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        elif cfg.base_loss == "focal":
            self.criterion = FocalLoss(gamma=2.0, reduction="mean")

        if cfg.ral_initial_alpha > 0.0 or cfg.ral_final_alpha > 0.0:
            self.use_RAL = True
            self.radius_acc_loss = RadiusAccuracyLoss()
            self.ral_initial_alpha = cfg.ral_initial_alpha
            self.ral_alpha = self.ral_initial_alpha
            self.ral_final_alpha = cfg.ral_final_alpha
            self.final_epoch = cfg.num_epochs
        else:
            self.ral_alpha = 0.0
            self.use_RAL = False
            self.radius_acc_loss = None

        # test_calibration is initally set to False for the first test run
        # it will be overwritten to True for the second test run
        self.test_calibration = False
        self.ece_batch_size = 2**13

    def on_fit_start(self):
        self.print(f"Training for {self.cfg.num_epochs} epochs")
        self.print(f"Using radius accuracy loss with alpha = {self.ral_alpha}")
        self.print(f"Using {self.cfg.lr_scheduler} scheduler")

    def forward(self, x, size=None):
        return self.model(x, size=size)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.use_RAL:
            logits, radii = self.forward(x, size=y.shape[-2:])

            if self.ral_initial_alpha != self.ral_final_alpha:
                # Compute decayed/increased alpha
                diff = self.ral_initial_alpha - self.ral_final_alpha
                curr_epoch = min(self.current_epoch, self.final_epoch)
                self.ral_alpha = (
                    self.ral_initial_alpha - (curr_epoch / self.final_epoch) * diff
                )

            ral = self.radius_acc_loss(logits, y, radii) * self.ral_alpha
            ce_loss = self.criterion(logits, y)
            loss = ce_loss + ral
        else:
            logits, _ = self.forward(x, size=y.shape[-2:])
            loss = ce_loss = self.criterion(logits, y)
            ral = torch.zeros_like(loss)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("train/ce_loss", ce_loss, on_step=True, on_epoch=True, sync_dist=True)

        if self.use_RAL:
            self.log(
                "train/radius_loss", ral, on_step=True, on_epoch=True, sync_dist=True
            )

        lr = self.lr_schedulers().get_last_lr()[0]
        self.log("lr", lr, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def inference(self, image, label):
        size = label.shape[-2:]
        logits, _ = self.forward(image, size=size)
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=True)

        preds = F.softmax(logits, dim=1)
        return preds, logits

    def on_validation_start(self):
        self.radii = []
        self.ece = np.array([])
        self.intersections = np.array([])
        self.unions = np.array([])
        self.targets = np.array([])

    def validation_forward(self, batch, tau=None):
        x, y = batch
        if len(y.shape) == 4:
            y = y.permute(0, 3, 1, 2).squeeze(1)

        size = y.shape[-2:]
        logits, _ = self.forward(x, size=size)
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=True)
        preds = F.softmax(logits, dim=1)
        output = preds.max(1)[1]

        if tau is not None:
            logits = logits / tau

        intersection, union, target = intersectionAndUnionGPU(
            output, y, self.num_classes, ignore_index=255
        )

        intersection = np.expand_dims(intersection, axis=0)
        union = np.expand_dims(union, axis=0)
        target = np.expand_dims(target, axis=0)

        if self.intersections.size == 0:
            self.intersections = intersection
            self.unions = union
            self.targets = target
        else:
            self.intersections = np.concatenate((self.intersections, intersection), axis=0)
            self.unions = np.concatenate((self.unions, union), axis=0)
            self.targets = np.concatenate((self.targets, target), axis=0)

        loss = self.criterion(logits, y)

        # flatten the logits and labels
        logits = (
            logits.view(logits.shape[0], logits.shape[1], -1)
            .permute(0, 2, 1)
            .reshape(-1, logits.shape[1])
        )
        y = y.view(-1, 1)

        # divide into batches of m pixels to avoid memory issues
        m = self.ece_batch_size
        n_batches = logits.shape[0] // m
        for j in range(n_batches):
            logits_batch = logits[j * m : (j + 1) * m]
            y_batch = y[j * m : (j + 1) * m]
            ece_i = calculate_ece(logits_batch, y_batch, n_bins=15)
            self.ece = np.append(self.ece, ece_i)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.validation_forward(batch)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_end(self):
        # gather all the metrics across all the processes
        intersections = self.all_gather(self.intersections)
        unions = self.all_gather(self.unions)
        targets = self.all_gather(self.targets)
        eces = self.all_gather(self.ece)

        intersections = intersections.flatten(0, 1)
        unions = unions.flatten(0, 1)
        targets = targets.flatten(0, 1)
        eces = eces.flatten()

        # calculate the final mean iou and accuracy
        intersections = intersections.sum(axis=0)
        unions = unions.sum(axis=0)
        targets = targets.sum(axis=0)

        iou_class = intersections / (unions + 1e-10)
        accuracy_class = intersections / (targets + 1e-10)

        mIoU = iou_class.mean() * 100
        mAcc = accuracy_class.mean() * 100
        aAcc = intersections.sum() / (targets.sum() + 1e-10) * 100

        ece = eces.mean()

        return mIoU, mAcc, aAcc, ece

    def on_validation_epoch_end(self):
        mIoU, mAcc, aAcc, ece = self.validation_end()
        self.log("val/ece", ece, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val/mIoU", mIoU, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val/mAcc", mAcc, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log("val/aAcc", aAcc, on_epoch=True, sync_dist=True, prog_bar=False)
        self.print(f"\nEpoch: {self.current_epoch}, ECE: {ece:.2f}\n")
        self.log("val_mIoU", mIoU, on_epoch=True, sync_dist=True, prog_bar=False)

    def on_test_start(self):
        self.on_validation_start()
        self.tau = None

        if self.test_calibration:
            # self.print("Computing optimal tau...")
            # self.tau = get_optimal_confidence_tau(
            #     self.model, self.trainer.test_dataloaders, self.criterion
            # )
            # self.print(f"Optimal tau: {self.tau}")
            self.tau = 2.0

    def test_step(self, batch, batch_idx):
        loss = self.validation_forward(batch, tau=self.tau)
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self):
        mIoU, mAcc, aAcc, ece = self.validation_end()
        self.log("test/ece", ece, on_epoch=True, sync_dist=True)
        self.log("test/mIoU", mIoU, on_epoch=True, sync_dist=True)
        # self.log("test/mAcc", mAcc, on_epoch=True, sync_dist=True)
        # self.log("test/aAcc", aAcc, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        if self.cfg.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.lr,
                momentum=0.9,
                weight_decay=self.cfg.weight_decay,
                nesterov=True,
            )
        elif self.cfg.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer == "RiemannianSGD":
            optimizer = RiemannianSGD(
                self.model.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                momentum=0.9,
                nesterov=True,
                stabilize=1,
            )
        elif self.cfg.optimizer == "RiemannianAdam":
            optimizer = RiemannianAdam(
                self.model.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                stabilize=1,
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.cfg.optimizer}")

        lr_scheduler = None
        if self.cfg.use_lr_scheduler:
            scheduler_type = self.cfg.lr_scheduler

            if scheduler_type == "MultiStepLR":
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=self.cfg.lr_scheduler_milestones,
                    gamma=self.cfg.lr_scheduler_gamma,
                )
            elif scheduler_type == "CosineAnnealingLR":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.cfg.num_epochs,
                    eta_min=self.cfg.lr * 0.01,
                )
            else:
                raise ValueError(f"Invalid scheduler type: {scheduler_type}")

        return [optimizer], [lr_scheduler]
