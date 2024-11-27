import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from train_learner import TrainLearner
from utils.initialize import get_config, select_dataset


def main(cfg):
    pl.seed_everything(cfg.seed)

    train_loader, val_loader, test_loader, _, _ = select_dataset(
        cfg, validation_split=cfg.validation_split
    )

    model = TrainLearner(cfg)

    if cfg.load_checkpoint:
        checkpoint = torch.load(cfg.load_checkpoint)
        model.load_state_dict(checkpoint["model"])

    logger = None
    if cfg.wandb:
        logger = WandbLogger(
            project="CPHNN",
            name=cfg.exp_name,
            group=cfg.dataset,
            entity="pinlab-sapienza",
            config=cfg,
            notes=cfg.notes,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.output_dir,
        filename="best_epoch-{epoch}_val_mIoU-{val_mIoU:.2f}",
        save_top_k=1,
        monitor="val_mIoU",
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.num_epochs,
        accelerator="gpu",
        devices=cfg.gpus,
        logger=logger,
        callbacks=[checkpoint_callback],
        precision=32 if cfg.dtype == "float32" else 64,
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)


if __name__ == "__main__":
    cfg = get_config()

    if cfg.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif cfg.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise "Wrong dtype in configuration -> " + cfg.dtype

    if cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)

    main(cfg)
