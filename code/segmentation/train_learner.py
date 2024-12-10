import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lib.geoopt.optim import RiemannianAdam, RiemannianSGD
from losses.focal_loss import FocalLoss
from losses.radius_loss import RadiusAccuracyLoss
from torchmetrics.classification import MulticlassJaccardIndex
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

    def on_train_start(self):
        self.print("\nModel initialized with the following configuration:")
        self.print(self.cfg)
        self.print("\n")

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
        self.mIoU_metric = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=255,
            average="micro",
        ).to(self.device)

    def validation_forward(self, batch, tau=None, mode="val"):
        x, y = batch
        if len(y.shape) == 4:
            y = y.permute(0, 3, 1, 2).squeeze(1)

        preds, logits = self.inference(x, y)
        preds = F.softmax(logits, dim=1)
        output = preds.argmax(dim=1)

        # calculate mIoU
        miou = self.mIoU_metric(output, y) * 100
        self.log(f"{mode}/mIoU", miou, on_step=False, on_epoch=True, sync_dist=True)
        if mode == "val":
            self.log("val_mIoU", miou, on_epoch=True, sync_dist=True, prog_bar=False)

        # calculate loss
        if mode == "val":
            loss = self.criterion(logits, y)
            self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        # calculate ECE
        # flatten the logits and labels
        logits = logits.view(-1, self.num_classes)
        y = y.view(-1)
        ece = calculate_ece(logits, y, n_bins=15)
        self.log(f"{mode}/ece", ece, on_step=False, on_epoch=True, sync_dist=True)

        if tau is not None:
            logits = logits / tau
            ece_ts = calculate_ece(logits, y, n_bins=15)
            self.log(f"{mode}/ece_ts", ece_ts, on_step=False, on_epoch=True, sync_dist=True)
            

    def validation_step(self, batch, batch_idx):
        self.validation_forward(batch, tau=None)

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
        self.validation_forward(batch, tau=self.tau, mode="test")

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
