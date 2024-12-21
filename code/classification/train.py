"""_summary_

Returns:
    _type_: _description_
"""

# Change working directory to parent HyperbolicCV/code
import os
import random
import sys

working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)
# -----------------------------------------------------

import random

import configargparse
import numpy as np
import torch
import wandb
from classification.losses.focal_loss import FocalLoss
from classification.losses.radius_label_smoothing import RadiusLabelSmoothing
from classification.losses.radius_loss import RadiusAccuracyLoss, RadiusConfidenceLoss #, WeightedRadiusAccuracyLoss
from classification.utils.calibration_metrics import CalibrationMetrics
from classification.utils.initialize import (
    get_config,
    load_checkpoint,
    select_dataset,
    select_model,
    select_optimizer,
)
from lib.utils.utils import AverageMeter, accuracy
from lightning.fabric import Fabric
from torch.nn import DataParallel
from tqdm import tqdm


def main(cfg):
    fabric = Fabric(accelerator="gpu", devices=cfg.gpus)
    device = fabric.device
    fabric.launch()

    if len(cfg.gpus) > 1:
        master_process = fabric.global_rank == 0
    else:
        master_process = True

    if cfg.output_dir and master_process:
        if not os.path.exists(cfg.output_dir):
            print("Create missing output directory...")
            os.mkdir(cfg.output_dir)

    if cfg.wandb and master_process:
        wandb.init(
            entity="pinlab-sapienza",
            project="CPHNN",
            group=cfg.dataset,
            name=cfg.exp_name,
            config=cfg,
            notes=cfg.notes,
        )

    # device = args.device[0]
    torch.cuda.empty_cache()

    fabric.print("Running experiment: " + cfg.exp_name)
    fabric.print("Arguments:")
    fabric.print(cfg)

    fabric.print(f"Loading dataset with validation_split = {cfg.validation_split}...")
    train_loader, val_loader, test_loader, img_dim, num_classes = select_dataset(
        cfg, validation_split=cfg.validation_split
    )

    fabric.print("Creating model...")
    model = select_model(img_dim, num_classes, cfg)
    model = model.to(device)
    # model = DataParallel(model, device_ids=args.device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fabric.print(
        f"-> Number of model params: {num_params} (trainable: {num_trainable_params})"
    )

    fabric.print("Creating optimizer...")
    optimizer, lr_scheduler = select_optimizer(model, cfg)

    fabric.print("Setup Fabric")
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader, test_loader = fabric.setup_dataloaders(
        train_loader, val_loader, test_loader
    )

    if cfg.base_loss == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif cfg.base_loss == "focal":
        criterion = FocalLoss(gamma=2.0, reduction="mean")

    if cfg.ral_initial_alpha > 0.0 or cfg.ral_final_alpha > 0.0:
        use_RAL = True
        radius_acc_loss = RadiusAccuracyLoss()
        # radius_acc_loss = WeightedRadiusAccuracyLoss()
        ral_initial_alpha = cfg.ral_initial_alpha
        ral_alpha = ral_initial_alpha
        ral_final_alpha = cfg.ral_final_alpha
        final_epoch = cfg.num_epochs
    else:
        ral_alpha = 0.0
        use_RAL = False
        radius_acc_loss = None

    rcl_alpha = cfg.radius_conf_loss
    if rcl_alpha > 0.0:
        use_RCL = True
        radius_conf_loss = RadiusConfidenceLoss(n_bins=15)
    else:
        use_RCL = False
        radius_conf_loss = None

    if cfg.radius_label_smoothing:
        radius_label_smoothing = RadiusLabelSmoothing(
            device,
            n_classes=num_classes,
            ema_alpha=cfg.ema_alpha,
            manifold=cfg.decoder_manifold,
        )
    else:
        radius_label_smoothing = None
    fabric.print(f"Using radius accuracy loss with alpha = {ral_alpha}")
    fabric.print(f"Using radius confidence loss with alpha = {rcl_alpha}")
    fabric.print(f"Using radius label smoothing = {cfg.radius_label_smoothing}")

    if use_RAL or use_RCL or radius_label_smoothing:
        use_radius_loss = True
    else:
        use_radius_loss = False

    start_epoch = 0
    if cfg.load_checkpoint:
        fabric.print(f"Loading model checkpoint from {cfg.load_checkpoint}")
        model, optimizer, lr_scheduler, start_epoch = load_checkpoint(
            model, optimizer, lr_scheduler, cfg
        )

    fabric.print("Training...")
    global_step = start_epoch * len(train_loader)

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        losses = AverageMeter("Loss", ":.4e")
        radius_acc_losses = AverageMeter("RadiusAccuracyLoss", ":.4e")
        radius_conf_losses = AverageMeter("RadiusConfidenceLoss", ":.4e")
        ce_losses = AverageMeter("CELoss", ":.4e")
        acc1 = AverageMeter("Acc@1", ":6.2f")
        acc5 = AverageMeter("Acc@5", ":6.2f")

        for batch_idx, (x, y) in tqdm(
            enumerate(train_loader), total=len(train_loader), disable=not master_process
        ):
            # ------- Start iteration -------
            x, y = x.to(device), y.to(device)

            if use_radius_loss:
                embeds = model.module.embed(x)
                logits = model.module.decoder(embeds)
                radii = torch.norm(embeds, dim=-1)

                ce_y = y
                if cfg.radius_label_smoothing:
                    smoothed_y = radius_label_smoothing(y, radii)
                    ce_y = smoothed_y

                ce_loss = criterion(logits, ce_y)

                if use_RAL:
                    if ral_initial_alpha != ral_final_alpha:
                        # Compute decayed/increased alpha
                        diff = ral_initial_alpha - ral_final_alpha
                        curr_epoch = min(epoch, final_epoch)
                        ral_alpha = ral_initial_alpha - (curr_epoch / final_epoch) * diff
                    ral = radius_acc_loss(logits, y, radii) * ral_alpha
                else:
                    ral = torch.zeros_like(ce_loss)

                if use_RCL:
                    rcl = radius_conf_loss(logits, y, radii) * rcl_alpha
                else:
                    rcl = torch.zeros_like(ce_loss)

                loss = ce_loss + ral + rcl
            else:
                logits = model(x)
                loss = ce_loss = criterion(logits, y)  # Compute loss
                ral, rcl = torch.zeros_like(ce_loss), torch.zeros_like(ce_loss)

            # check if loss is nan and exit
            if torch.isnan(loss).any():
                print(f"NaN detected in loss. Skipping batch {batch_idx}.")
                exit()

            optimizer.zero_grad()
            # loss.backward()
            fabric.backward(loss)
            optimizer.step()

            with torch.no_grad():
                top1, top5 = accuracy(logits, y, topk=(1, 5))
                losses.update(loss.item())
                ce_losses.update(ce_loss.item())
                radius_acc_losses.update(ral.item())
                radius_conf_losses.update(rcl.item())
                acc1.update(top1.item())
                acc5.update(top5.item())

            global_step += 1
            # ------- End iteration -------

        # ------- Start validation and logging -------
        with torch.no_grad():
            if lr_scheduler is not None:
                if (epoch + 1) == cfg.lr_scheduler_milestones[
                    0
                ]:  # skip the first drop for some Parameters
                    optimizer.param_groups[1]["lr"] *= (
                        1 / cfg.lr_scheduler_gamma
                    )  # Manifold params
                    fabric.print("Skipped lr drop for manifold parameters")

                lr_scheduler.step()

            loss_val, acc1_val, acc5_val, cm = evaluate(
                model,
                val_loader,
                criterion,
                device,
                num_classes,
                manifold=cfg.decoder_manifold,
                fabric=fabric,
            )

            log_string = f"Epoch {epoch + 1}/{cfg.num_epochs}: "
            log_string += f"Loss={losses.avg:.4f}, "
            if use_RAL:
                log_string += f"RadiusAccLoss={radius_acc_losses.avg:.4f}, "
            if use_RCL:
                log_string += f"RadiusConfLoss={radius_conf_losses.avg:.4f}, "
            log_string += f"Acc@1={acc1.avg:.4f}, Acc@5={acc5.avg:.4f}, "
            log_string += f"Validation: Loss={loss_val:.4f}, "
            log_string += f"Acc@1={acc1_val:.4f}, Acc@5={acc5_val:.4f}"
            fabric.print(log_string)

            # if args.radius_label_smoothing:
            #     print("EMA of radii per class:")
            #     print(radius_label_smoothing.radii_ema.tolist())
            #     print("\n")

            if cfg.wandb and master_process:
                log_dict = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train/loss": losses.avg,
                    "train/ce_loss": ce_losses.avg,
                    "train/radius_acc_loss": radius_acc_losses.avg,
                    "train/radius_conf_loss": radius_conf_losses.avg,
                    "train/acc1": acc1.avg,
                    "train/acc5": acc5.avg,
                    "mce": cm["mce"],
                    "ece": cm["ece"],
                    "rmsce": cm["rmsce"],
                    "val/loss": loss_val,
                    "val/acc1": acc1_val,
                    "val/acc5": acc5_val,
                }
                if use_RAL:
                    log_dict["train/ral_alpha"] = ral_alpha
                wandb.log(log_dict)

            # Testing for best model
            if acc1_val > best_acc:
                best_acc = acc1_val
                best_epoch = epoch + 1
                if cfg.output_dir and master_process:
                    save_path = f"{cfg.output_dir}/best_{cfg.exp_name}.pth"
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": (
                                lr_scheduler.state_dict() if lr_scheduler else None
                            ),
                            "epoch": epoch,
                            "args": cfg,
                        },
                        save_path,
                    )
        # ------- End validation and logging -------

    fabric.print("-----------------\nTraining finished\n-----------------")
    fabric.print("Best epoch = {}, with Acc@1={:.4f}".format(best_epoch, best_acc))

    if cfg.output_dir and master_process:
        save_path = f"{cfg.output_dir}/final_{cfg.exp_name}.pth"
        torch.save(
            {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
                "epoch": epoch,
                "args": cfg,
            },
            save_path,
        )
        fabric.print(f"Model saved to {save_path}")
    else:
        fabric.print("Model not saved.")

    print("Testing final model...")
    loss_test, acc1_test, acc5_test, cm = evaluate(
        model,
        test_loader,
        criterion,
        device,
        num_classes,
        manifold=cfg.decoder_manifold,
        fabric=fabric,
    )

    fabric.print(
        "Results: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
            loss_test, acc1_test, acc5_test
        )
    )

    fabric.print("Testing best model...")
    if cfg.output_dir and master_process:
        print("Loading best model...")
        save_path = f"{cfg.output_dir}/best_{cfg.exp_name}.pth"
        checkpoint = torch.load(save_path, map_location=device)
        model.module.load_state_dict(checkpoint["model"], strict=True)

        loss_test, acc1_test, acc5_test, cm = evaluate(
            model,
            test_loader,
            criterion,
            device,
            num_classes,
            manifold=cfg.decoder_manifold,
            fabric=fabric,
        )

        print(
            f"Results: Loss={loss_test:.4f}, Acc@1={acc1_test:.4f}, Acc@5={acc5_test:.4f}"
        )

        if cfg.wandb:
            wandb.log(
                {
                    "test/loss": loss_test,
                    "test/acc1": acc1_test,
                    "test/acc5": acc5_test,
                    "mce": cm["mce"],
                    "ece": cm["ece"],
                    "rmsce": cm["rmsce"],
                }
            )
    else:
        fabric.print("Best model not saved, because no output_dir given.")


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    num_classes,
    calibration=None,
    tau=None,
    manifold="euclidean",
    fabric=None,
):
    """Evaluates model performance"""
    model.eval()
    model.to(device)

    losses = AverageMeter("Loss", ":.4e")
    acc1 = AverageMeter("Acc@1", ":6.2f")
    acc5 = AverageMeter("Acc@5", ":6.2f")

    radii = []
    cm = CalibrationMetrics(n_classes=num_classes)

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        embeds = model.module.embed(x)
        if calibration == "radius" and tau:
            embeds = embeds / tau

        logits = model.module.decoder(embeds)
        if calibration == "confidence" and tau:
            logits = logits / tau

        if manifold == "euclidean":
            radius = torch.norm(embeds, dim=-1)
        else:
            # radius = model.module.dec_manifold.dist0(embeds)
            radius = torch.norm(embeds, dim=-1)
        radii.append(radius.cpu().numpy())

        logits = torch.nn.functional.softmax(logits, dim=-1)

        loss = criterion(logits, y)

        top1, top5 = accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item())
        acc1.update(top1.item(), x.shape[0])
        acc5.update(top5.item(), x.shape[0])
        cm.update(logits, y)

    calib_metrics = cm.compute()
    if fabric:
        fabric.print("\n===== Calibration metrics ===== \n")
    else:
        print("\n===== Calibration metrics ===== \n")
    for k, v in calib_metrics.items():
        if fabric:
            fabric.print(f"{k.upper()}: {round(v, 4)}")
        else:
            print(f"{k.upper()}: {round(v, 4)}")
    if fabric:
        fabric.print("\n=============================== \n")
    else:
        print("\n=============================== \n")

    radii = np.concatenate(radii)
    avg_radius = np.mean(radii)
    if fabric:
        fabric.print(f"Average norm: {avg_radius}")
    else:
        print(f"Average norm: {avg_radius}")

    return losses.avg, acc1.avg, acc5.avg, calib_metrics


# ----------------------------------
if __name__ == "__main__":
    cfg = get_config()

    if cfg.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif cfg.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise "Wrong dtype in configuration -> " + cfg.dtype

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    main(cfg)
