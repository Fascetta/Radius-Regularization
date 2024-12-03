import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lib.geoopt.optim import RiemannianAdam, RiemannianSGD
from losses.focal_loss import FocalLoss
from losses.radius_loss import RadiusAccuracyLoss
from torchmetrics import Accuracy
from utils.initialize import select_dataset, select_model
from utils.metrics import CalibrationMetrics, calculate_ece, intersectionAndUnionGPU


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

        print(f"Using radius accuracy loss with alpha = {self.ral_alpha}")

        # evaluation metrics
        # self.intersections = np.array([])
        # self.unions = np.array([])
        # self.targets = np.array([])

    def forward(self, x, size=None):
        return self.model(x, size=size)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.use_RAL:
            logits, radii = self.forward(x, size=y.shape[-2:])
            # radii = torch.norm(embeds, dim=1)

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

    def inference(
        self,
        image,
        label,
        flip=True,
    ):
        size = label.shape[-2:]
        if flip:
            image = torch.cat([image, torch.flip(image, [3])], 0)

        output, _ = self.forward(image, size=size)
        output = F.interpolate(output, size=size, mode="bilinear", align_corners=True)

        logits = output
        output = F.softmax(output, dim=1)
        if flip:
            output = (output[0] + output[1].flip(2)) / 2
        else:
            output = output[0]
        return output.unsqueeze(0), logits

    def validation_forward(self, batch):
        x, y = batch
        y = y.permute(0, 3, 1, 2).squeeze(1)

        preds, logits = self.inference(x, y, flip=False)
        output = preds.max(1)[1]

        intersection, union, target = intersectionAndUnionGPU(
            output, y, self.num_classes, ignore_index=255
        )

        miou = (intersection / (union + 1e-10)) * 100
        miou = miou.mean()
        loss = self.criterion(logits, y)

        # flatten the logits and labels
        logits = (
            logits.view(logits.shape[0], logits.shape[1], -1)
            .permute(0, 2, 1)
            .reshape(-1, logits.shape[1])
        )
        y = y.view(-1, 1)

        # divide into batches of m pixels to avoid memory issues
        m = 2**13
        n_batches = logits.shape[0] // m
        for j in range(n_batches):
            logits_batch = logits[j * m : (j + 1) * m]
            y_batch = y[j * m : (j + 1) * m]
            ece_i = calculate_ece(logits_batch, y_batch, n_bins=15)
            self.ece.append(ece_i)

        return loss, miou

    def on_validation_start(self):
        self.radii = []
        self.ece = []
        # self.cm = CalibrationMetrics(n_classes=self.num_classes)

    def validation_step(self, batch, batch_idx):
        loss, miou = self.validation_forward(batch)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mIoU", miou, on_epoch=True, prog_bar=True, sync_dist=True)

        # for checkpoint saving
        self.log("val_mIoU", miou, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        ece = np.mean(self.ece)
        self.log("val/ece", ece, on_epoch=True, sync_dist=True, prog_bar=True)
        self.print(f"\nEpoch: {self.current_epoch}, ECE: {ece:.2f}\n")

    def on_test_start(self):
        self.on_validation_start()

    def test_step(self, batch, batch_idx):
        loss, miou = self.validation_forward(batch)

        self.log("test/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/mIoU", miou, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        ece = np.mean(self.ece)
        self.log("test/ece", ece, on_epoch=True, sync_dist=True, prog_bar=True)

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
            print(f"Using {scheduler_type} scheduler")

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
