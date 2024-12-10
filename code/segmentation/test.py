# -----------------------------------------------------
# Change working directory to parent HyperbolicCV/code
import os
import sys

# lib_path = os.path.join(working_dir)
# sys.path.append(lib_path)
# -----------------------------------------------------
import pytorch_lightning as pl
import torch
from train_learner import TrainLearner
from utils.initialize import get_config, select_dataset
from yacs.config import CfgNode


# working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
# os.chdir(working_dir)


def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)

    _, _, test_loader, _, _ = select_dataset(cfg, validation_split=cfg.validation_split)

    # load the trained model and its hyperparameters
    ckpt_cfg = torch.load(cfg.checkpoint_path, map_location="cpu")["hyper_parameters"]
    ckpt_cfg = CfgNode(ckpt_cfg)
    model = TrainLearner(cfg)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.gpus,
        logger=False,
        precision="16-mixed",
    )

    print("\nTesting the following model:")
    print("="*30, f"{ckpt_cfg.exp_name}", "="*30)
    print()

    model.test_calibration = True
    trainer.test(model, test_loader, cfg.checkpoint_path)


if __name__ == "__main__":
    cfg = get_config()

    main(cfg)
