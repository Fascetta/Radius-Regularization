"""_summary_

Returns:
    _type_: _description_
"""

# Change working directory to parent HyperbolicCV/code
import os
import random
import sys
import copy

import configargparse
import learn2learn as l2l
import numpy as np
import torch
from torch.nn import DataParallel
from tqdm import tqdm

import wandb

working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)
# -----------------------------------------------------

from classification.utils.calibration_metrics import CalibrationMetrics
from classification.utils.focal_loss import FocalLoss
from classification.utils.initialize import (load_checkpoint, select_dataset,
                                             select_model, select_optimizer)
from classification.utils.radius_loss import (RadiusAccuracyLoss,
                                              RadiusConfidenceLoss)
from classification.utils.dECE import differentiable_ece
from lib.utils.utils import AverageMeter, accuracy


def get_arguments():
    """Parses command-line options."""
    parser = configargparse.ArgumentParser(
        description="Image classification training", add_help=True
    )

    parser.add_argument(
        "-c",
        "--config_file",
        required=False,
        default=None,
        is_config_file=True,
        type=str,
        help="Path to config file.",
    )

    # Output settings
    parser.add_argument(
        "--exp_name", default="test", type=str, help="Name of the experiment."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Path for output files (relative to working directory).",
    )

    # General settings
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=lambda s: [str(item) for item in s.replace(" ", "").split(",")],
        help="List of devices split by comma, can also be a single device or 'cpu')",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        type=str,
        choices=["float32", "float64"],
        help="Set floating point precision.",
    )
    parser.add_argument(
        "--seed", default=1, type=int, help="Set seed for deterministic training."
    )
    parser.add_argument(
        "--load_checkpoint",
        default=None,
        type=str,
        help="Path to model checkpoint (weights, optimizer, epoch).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model for faster inference (requires PyTorch 2).",
    )

    # General training parameters
    parser.add_argument(
        "--num_epochs", default=200, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Training batch size."
    )
    parser.add_argument(
        "--lr", default=1e-1, type=float, help="Training learning rate."
    )
    parser.add_argument(
        "--weight_decay",
        default=5e-4,
        type=float,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--optimizer",
        default="RiemannianSGD",
        type=str,
        choices=["RiemannianAdam", "RiemannianSGD", "Adam", "SGD", "AdamW"],
        help="Optimizer for training.",
    )
    parser.add_argument(
        "--use_lr_scheduler",
        action="store_true",
        help="If learning rate should be reduced after step epochs using a LR scheduler.",
    )
    parser.add_argument(
        "--lr_scheduler",
        default="MultiStepLR",
        type=str,
        choices=["MultiStepLR", "CosineAnnealingLR"],
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_scheduler_milestones",
        default=[60, 120, 160],
        type=int,
        nargs="+",
        help="Milestones of LR scheduler.",
    )
    parser.add_argument(
        "--lr_scheduler_gamma",
        default=0.2,
        type=float,
        help="Gamma parameter of LR scheduler.",
    )
    parser.add_argument(
        "--base_loss",
        default="cross_entropy",
        type=str,
        choices=["cross_entropy", "focal"],
        help="Base loss function for training.",
    )

    # General validation/testing hyperparameters
    parser.add_argument(
        "--batch_size_test",
        default=128,
        type=int,
        help="Validation/Testing batch size.",
    )
    parser.add_argument(
        "--validation_split",
        action="store_true",
        help="Use validation split of training data.",
    )

    # Model selection
    parser.add_argument(
        "--num_layers",
        default=18,
        type=int,
        choices=[18, 50],
        help="Number of layers in ResNet.",
    )
    parser.add_argument(
        "--embedding_dim",
        default=512,
        type=int,
        help="Dimensionality of classification embedding space (could be expanded by ResNet)",
    )
    parser.add_argument(
        "--encoder_manifold",
        default="lorentz",
        type=str,
        choices=["euclidean", "lorentz"],
        help="Select conv model encoder manifold.",
    )
    parser.add_argument(
        "--decoder_manifold",
        default="lorentz",
        type=str,
        choices=["euclidean", "lorentz", "poincare"],
        help="Select conv model decoder manifold.",
    )

    # Hyperbolic geometry settings
    parser.add_argument(
        "--learn_k",
        action="store_true",
        help="Set a learnable curvature of hyperbolic geometry.",
    )
    parser.add_argument(
        "--encoder_k",
        default=1.0,
        type=float,
        help="Initial curvature of hyperbolic geometry in backbone (geoopt.K=-1/K).",
    )
    parser.add_argument(
        "--decoder_k",
        default=1.0,
        type=float,
        help="Initial curvature of hyperbolic geometry in decoder (geoopt.K=-1/K).",
    )
    parser.add_argument(
        "--clip_features",
        default=1.0,
        type=float,
        help="Clipping parameter for hybrid HNNs proposed by Guo et al. (2022)",
    )
    parser.add_argument(
        "--radius_acc_loss",
        default=0.0,
        type=float,
        help="Use radius accuracy loss together with cross-entropy loss.",
    )
    parser.add_argument(
        "--radius_conf_loss",
        default=0.0,
        type=float,
        help="Use radius confidence loss together with cross-entropy loss.",
    )

    parser.add_argument(
        "--meta_lr",
        default=0.0,
        type=float,
        help="Use Meta-Learning.",
    )

    parser.add_argument(
    "--num_inner_steps",
    default=5,
    type=int,
    help="Define the number of steps in the inner loop of Meta-Learning",
    )

    parser.add_argument(
    "--ece_alpha", 
    default=0.0,
    type=float,
    help="Weight for ECE loss in total loss."
    )

    # Dataset settings
    parser.add_argument(
        "--dataset",
        default="CIFAR-100",
        type=str,
        choices=["MNIST", "CIFAR-10", "CIFAR-100", "Tiny-ImageNet"],
        help="Select a dataset.",
    )

    # Logging settings
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases for logging.",
    )

    parsed_args = parser.parse_args()

    parsed_args.output_dir = os.path.join(
        parsed_args.output_dir, str(parsed_args.dataset).lower()
    )

    return parsed_args


def main(args):
    if args.wandb:
        wandb.init(
            entity="pinlab-sapienza",
            project="CPHNN",
            group=args.dataset,
            name=args.exp_name,
            config=args,
        )
    device = args.device[0]
    torch.cuda.empty_cache()

    print("Running experiment: " + args.exp_name)
    print("Arguments:")
    print(args)

    print(f"Loading dataset with validation_split = {args.validation_split}...")
    train_loader, val_loader, test_loader, img_dim, num_classes = select_dataset(
        args, validation_split=args.validation_split
    )

    print("Creating model...")
    model = select_model(img_dim, num_classes, args)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"-> Number of model params: {num_params} (trainable: {num_trainable_params})"
    )

    print("Creating optimizer...")
    optimizer, lr_scheduler = select_optimizer(model, args)


    if args.base_loss == "focal":
        criterion = FocalLoss(gamma=2.0, reduction="mean")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    ral_alpha = args.radius_acc_loss
    rcl_alpha = args.radius_conf_loss
    radius_acc_loss = RadiusAccuracyLoss()
    radius_conf_loss = RadiusConfidenceLoss(n_bins=15)
    print(f"Using radius accuracy loss with alpha = {ral_alpha}")
    print(f"Using radius confidence loss with alpha = {rcl_alpha}")

    use_radius_loss = False
    if ral_alpha > 0.0 or rcl_alpha > 0.0:
        use_radius_loss = True

    start_epoch = 0
    if args.load_checkpoint:
        print(f"Loading model checkpoint from {args.load_checkpoint}")
        model, optimizer, lr_scheduler, start_epoch = load_checkpoint(
            model, optimizer, lr_scheduler, args
        )

    model = DataParallel(model, device_ids=args.device)

    if args.meta_lr:
        model = l2l.algorithms.MAML(model, lr=args.meta_lr, first_order=False)

        def inner_loop(model, x, y, criterion, args):
            learner = copy.deepcopy(model)  # Clone model for inner-loop updates
            inner_optimizer = torch.optim.SGD(learner.parameters(), lr=args.lr)
            for _ in range(args.num_inner_steps):  # Inner-loop steps
                logits = learner(x)
                inner_loss = criterion(logits, y)
                inner_optimizer.zero_grad()
                inner_loss.backward()
                inner_optimizer.step()
            return learner  # Return adapted model

    print("Training...")
    global_step = start_epoch * len(train_loader)

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        losses = AverageMeter("Loss", ":.4e")
        radius_acc_losses = AverageMeter("RadiusAccuracyLoss", ":.4e")
        radius_conf_losses = AverageMeter("RadiusConfidenceLoss", ":.4e")
        ce_losses = AverageMeter("CELoss", ":.4e")
        acc1 = AverageMeter("Acc@1", ":6.2f")
        acc5 = AverageMeter("Acc@5", ":6.2f")

        for _, (x, y) in tqdm(enumerate(train_loader)):
            # ------- Start iteration -------
            x, y = x.to(device), y.to(device)

            if args.meta_lr:
                # Meta-learning inner-loop adaptation

                adapted_model = inner_loop(model, x, y, criterion, args)
                logits = adapted_model(x)  # Use adapted model for outer-loop loss

                if use_radius_loss:
                    # Calculate losses including radius-based losses if specified
                    embeds = adapted_model.module.embed(x)
                    logits = adapted_model.module.decoder(embeds)
                    if args.decoder_manifold == "euclidean":
                        radii = torch.norm(embeds, dim=-1)
                    else:
                        radii = adapted_model.module.dec_manifold.dist0(embeds)

                    ral = radius_acc_loss(logits, y, radii) * ral_alpha
                    rcl = radius_conf_loss(logits, y, radii) * rcl_alpha
                    ce_loss = criterion(logits, y)

                    diff_ece = differentiable_ece(logits, y)
                    loss = ce_loss + ral + rcl + diff_ece
                else:
                    # Incorporate differentiable ECE
                    ce_loss = criterion(logits, y)
                    diff_ece = differentiable_ece(logits, y)
                    loss = ce_loss + diff_ece

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # Meta-learning outer-loop update

            else:
                # Standard training procedure without meta-learning
                if use_radius_loss:
                    embeds = model.module.embed(x)
                    logits = model.module.decoder(embeds)
                    if args.decoder_manifold == "euclidean":
                        radii = torch.norm(embeds, dim=-1)
                    else:
                        radii = model.module.dec_manifold.dist0(embeds)
                    ral = radius_acc_loss(logits, y, radii) * ral_alpha
                    rcl = radius_conf_loss(logits, y, radii) * rcl_alpha
                    ce_loss = criterion(logits, y)
                    loss = ce_loss + ral + rcl
                else:
                    logits = model(x)
                    ce_loss = criterion(logits, y)
                    loss = ce_loss  # Loss calculation without meta-learning

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                top1, top5 = accuracy(logits, y, topk=(1, 5))
                losses.update(loss.item())
                ce_losses.update(ce_loss.item())
                # radius_acc_losses.update(ral.item())
                # radius_conf_losses.update(rcl.item())
                acc1.update(top1.item())
                acc5.update(top5.item())

            global_step += 1
            # ------- End iteration -------

        # ------- Start validation and logging -------
        with torch.no_grad():
            if lr_scheduler is not None:
                if (epoch + 1) == args.lr_scheduler_milestones[
                    0
                ]:  # skip the first drop for some Parameters
                    optimizer.param_groups[1]["lr"] *= (
                        1 / args.lr_scheduler_gamma
                    )  # Manifold params
                    print("Skipped lr drop for manifold parameters")

                lr_scheduler.step()

            loss_val, acc1_val, acc5_val, cm = evaluate(
                model,
                val_loader,
                criterion,
                device,
                num_classes,
                manifold=args.decoder_manifold,
            )

            if not use_radius_loss:
                print(
                    f"Epoch {epoch + 1}/{args.num_epochs}: "
                    f"Loss={losses.avg:.4f}, "
                    f"Acc@1={acc1.avg:.4f}, Acc@5={acc5.avg:.4f}, "
                    f"Validation: Loss={loss_val:.4f}, "
                    f"Acc@1={acc1_val:.4f}, Acc@5={acc5_val:.4f}"
                    f"\n"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{args.num_epochs}: "
                    f"Loss={losses.avg:.4f} "
                    f"RadiusAccLoss={radius_acc_losses.avg:.4f}, "
                    f"RadiusConfLoss={radius_conf_losses.avg:.4f}, "
                    f"Acc@1={acc1.avg:.4f}, Acc@5={acc5.avg:.4f}, "
                    f"Validation: Loss={loss_val:.4f}, "
                    f"Acc@1={acc1_val:.4f}, Acc@5={acc5_val:.4f}"
                    f"\n"
                )

            if args.wandb:
                wandb.log(
                    {
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
                )

            # Testing for best model
            if acc1_val > best_acc:
                best_acc = acc1_val
                best_epoch = epoch + 1
                if args.output_dir is not None:
                    save_path = args.output_dir + "/best_" + args.exp_name + ".pth"
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": (
                                lr_scheduler.state_dict()
                                if lr_scheduler is not None
                                else None
                            ),
                            "epoch": epoch,
                            "args": args,
                        },
                        save_path,
                    )
                if args.output_dir:
                    save_path = f"{args.output_dir}/best_{args.exp_name}.pth"
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": (
                                lr_scheduler.state_dict() if lr_scheduler else None
                            ),
                            "epoch": epoch,
                            "args": args,
                        },
                        save_path,
                    )
        # ------- End validation and logging -------

    print("-----------------\nTraining finished\n-----------------")
    print(f"Best epoch = {best_epoch}, with Acc@1={best_acc:.4f}")

    if args.output_dir:
        save_path = f"{args.output_dir}/final_{args.exp_name}.pth"
        torch.save(
            {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
                "epoch": epoch,
                "args": args,
            },
            save_path,
        )
        print(f"Model saved to {save_path}")
    else:
        print("Model not saved.")

    print("Testing final model...")
    loss_test, acc1_test, acc5_test, cm = evaluate(
        model,
        test_loader,
        criterion,
        device,
        num_classes,
        manifold=args.decoder_manifold,
    )

    print(f"Results: Loss={loss_test:.4f}, Acc@1={acc1_test:.4f}, Acc@5={acc5_test:.4f}")

    print("Testing best model...")
    if args.output_dir:
        print("Loading best model...")
        save_path = f"{args.output_dir}/best_{args.exp_name}.pth"
        checkpoint = torch.load(save_path, map_location=device)
        model.module.load_state_dict(checkpoint["model"], strict=True)

        loss_test, acc1_test, acc5_test, cm = evaluate(
            model,
            test_loader,
            criterion,
            device,
            num_classes,
            manifold=args.decoder_manifold,
        )

        print(
            f"Results: Loss={loss_test:.4f}, Acc@1={acc1_test:.4f}, Acc@5={acc5_test:.4f}"
        )

        if args.wandb:
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
        print("Best model not saved, because no output_dir given.")


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
):
    """Evaluates model performance"""
    model.eval()
    model.to(device)

    losses = AverageMeter("Loss", ":.4e")
    acc1 = AverageMeter("Acc@1", ":6.2f")
    acc5 = AverageMeter("Acc@5", ":6.2f")

    radii = []
    cm = CalibrationMetrics(n_classes=num_classes)

    for _, (x, y) in enumerate(dataloader):
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
            radius = model.module.dec_manifold.dist0(embeds)
        radii.append(radius.cpu().numpy())

        logits = torch.nn.functional.softmax(logits, dim=-1)

        loss = criterion(logits, y)

        top1, top5 = accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item())
        acc1.update(top1.item(), x.shape[0])
        acc5.update(top5.item(), x.shape[0])
        cm.update(logits, y)

    calib_metrics = cm.compute()
    print("\n===== Calibration metrics ===== \n")
    for k, v in calib_metrics.items():
        print(f"{k.upper()}: {round(v, 4)}")
    print("\n=============================== \n")

    radii = np.concatenate(radii)
    avg_radius = np.mean(radii)
    print(f"Average norm: {avg_radius}")

    return losses.avg, acc1.avg, acc5.avg, calib_metrics


# ----------------------------------
if __name__ == "__main__":
    args = get_arguments()

    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif args.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise "Wrong dtype in configuration -> " + args.dtype

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            print("Create missing output directory...")
            os.mkdir(args.output_dir)

    main(args)
