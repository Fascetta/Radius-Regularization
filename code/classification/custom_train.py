# -----------------------------------------------------
# Change working directory to parent HyperbolicCV/code
import os
import sys

from utils.ece_meter import ECEMeter

working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)
# -----------------------------------------------------

import random

import configargparse
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from classification.utils.calibration_metrics import CalibrationMetrics
from classification.utils.initialize import (
    load_checkpoint,
    select_dataset,
    select_model,
    select_optimizer,
)
from lib.geoopt.manifolds.lorentz.math import dist0
from lib.utils.utils import AverageMeter, accuracy
from torch.nn import DataParallel
from tqdm import tqdm


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
        "--exp_name",
        default="L-ResNet18-WeightedRadiusLoss",
        type=str,
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--output_dir",
        default="classification/output/cifar-100",
        type=str,
        help="Path for output files (relative to working directory).",
    )

    # General settings
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=lambda s: [str(item) for item in s.replace(" ", "").split(",")],
        help="List of devices (e.g. cuda:0,cuda:1), can also be single device or 'cpu')",
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
        choices=["RiemannianAdam", "RiemannianSGD", "Adam", "SGD"],
        help="Optimizer for training.",
    )
    parser.add_argument(
        "--use_lr_scheduler",
        action="store_true",
        help="If learning rate should be reduced after step epochs using a LR scheduler.",
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
        "--radius_loss",
        default=0.0,
        type=float,
        help="Use radius focal loss together with cross-entropy loss.",
    )

    # Dataset settings
    parser.add_argument(
        "--dataset",
        default="CIFAR-100",
        type=str,
        choices=["MNIST", "CIFAR-10", "CIFAR-100", "Tiny-ImageNet"],
        help="Select a dataset.",
    )

    args = parser.parse_args()

    return args


misclassified_moving_average = []


def update_moving_average(new_value, moving_average, alpha=0.9):
    if not moving_average:
        moving_average.append(new_value)
    else:
        moving_average.append(alpha * moving_average[-1] + (1 - alpha) * new_value)


def main(args):
    wandb.init(
        entity="pinlab-sapienza",
        project="CPHNN",
        group=args.dataset,
        name=args.exp_name,
    )
    device = args.device[0]
    torch.cuda.empty_cache()
    print("Running experiment: " + args.exp_name)
    print("Arguments:")
    print(args)
    train_loader, val_loader, test_loader, img_dim, num_classes = select_dataset(
        args, validation_split=args.validation_split
    )
    print("Creating model...")
    model = select_model(img_dim, num_classes, args)
    model = model.to(device)
    print(
        "-> Number of model params: {} (trainable: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )
    print("Creating optimizer...")
    optimizer, lr_scheduler = select_optimizer(model, args)
    criterion = torch.nn.CrossEntropyLoss()
    start_epoch = 0
    if args.load_checkpoint is not None:
        print("Loading model checkpoint from {}".format(args.load_checkpoint))
        model, optimizer, lr_scheduler, start_epoch = load_checkpoint(
            model, optimizer, lr_scheduler, args
        )
    model = DataParallel(model, device_ids=args.device)
    print("Training...")
    global_step = start_epoch * len(train_loader)
    best_acc = 0.0
    best_epoch = 0
    ece_meter = ECEMeter()
    manifold_curvature = torch.tensor(1.0)
    rl_alpha = args.radius_loss

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        losses = AverageMeter("Loss", ":.4e")
        acc1 = AverageMeter("Acc@1", ":6.2f")
        acc5 = AverageMeter("Acc@5", ":6.2f")
        ece_meter.reset()

        for _, (x, y) in tqdm(enumerate(train_loader)):
            # ------- Start iteration -------
            x, y = x.to(device), y.to(device)
            logits = model(x)
            confidences, predictions = torch.max(torch.softmax(logits, dim=1), 1)
            embeddings = model.module.embed(x)

            incorrect_indices = (
                predictions != y
            )  # Identify indices where predictions are incorrect
            incorrect_confidences = confidences[incorrect_indices]
            incorrect_embeddings = embeddings[incorrect_indices]
            radii = dist0(incorrect_embeddings, k=manifold_curvature)
            weighted_rl = radii * incorrect_confidences
            mean_weighted_rl = torch.mean(weighted_rl)

            base_loss = criterion(logits, y)  # Compute loss
            loss = base_loss + mean_weighted_rl * rl_alpha

            optimizer.zero_grad()  # Reset gradients
            loss.backward()  # Back-Propagation
            optimizer.step()  # Update model parameters

            with torch.no_grad():
                num_misclassified = torch.sum(incorrect_indices).item()
                update_moving_average(num_misclassified, misclassified_moving_average)

                top1, top5 = accuracy(logits, y, topk=(1, 5))
                losses.update(loss.item())
                acc1.update(top1.item())
                acc5.update(top5.item())
                ece_meter.update(logits, y)

            global_step += 1

        ece_score = ece_meter.compute()
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
            loss_val, acc1_val, acc5_val = evaluate(
                model, val_loader, criterion, device
            )
            print(
                "Epoch {}/{}: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}, Validation: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
                    epoch + 1,
                    args.num_epochs,
                    losses.avg,
                    acc1.avg,
                    acc5.avg,
                    loss_val,
                    acc1_val,
                    acc5_val,
                )
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": losses.avg,
                    "train/acc1": acc1.avg,
                    "train/acc5": acc5.avg,
                    "ece": ece_score,
                    "misclassified_moving_avg": misclassified_moving_average[-1],
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
        # ------- End validation and logging -------

    print("-----------------\nTraining finished\n-----------------")
    print("Best epoch = {}, with Acc@1={:.4f}".format(best_epoch, best_acc))

    if args.output_dir is not None:
        save_path = args.output_dir + "/final_" + args.exp_name + ".pth"
        torch.save(
            {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": (
                    lr_scheduler.state_dict() if lr_scheduler is not None else None
                ),
                "epoch": epoch,
                "args": args,
            },
            save_path,
        )
        print("Model saved to " + save_path)
    else:
        print("Model not saved.")

    print("Testing final model...")
    loss_test, acc1_test, acc5_test = evaluate(model, test_loader, criterion, device)

    print(
        "Results: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
            loss_test, acc1_test, acc5_test
        )
    )

    print("Testing best model...")
    if args.output_dir is not None:
        print("Loading best model...")
        save_path = args.output_dir + "/best_" + args.exp_name + ".pth"
        checkpoint = torch.load(save_path, map_location=device)
        model.module.load_state_dict(checkpoint["model"], strict=True)

        loss_test, acc1_test, acc5_test = evaluate(
            model, test_loader, criterion, device
        )

        print(
            "Results: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
                loss_test, acc1_test, acc5_test
            )
        )

        wandb.log(
            {
                "test/loss": loss_test,
                "test/acc1": acc1_test,
                "test/acc5": acc5_test,
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
    calibration_metrics=False,
    tau=None,
):
    """Evaluates model performance"""
    model.eval()
    model.to(device)

    losses = AverageMeter("Loss", ":.4e")
    acc1 = AverageMeter("Acc@1", ":6.2f")
    acc5 = AverageMeter("Acc@5", ":6.2f")

    norms = []

    if calibration_metrics:
        num_classes = criterion.numel()
        print(f"Number of classes: {num_classes}")
        cm = CalibrationMetrics(n_classes=100)

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        embeds = model.module.embed(x)
        if tau:
            embeds = embeds / tau
        logits = model.module.decoder(embeds)

        norms.append(torch.norm(embeds, dim=-1, p=2).cpu().numpy())

        logits = torch.nn.functional.softmax(logits, dim=-1)

        loss = criterion(logits, y)

        top1, top5 = accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item())
        acc1.update(top1.item(), x.shape[0])
        acc5.update(top5.item(), x.shape[0])

        if calibration_metrics:
            cm.update(logits, y)

    if calibration_metrics:
        calib_metrics = cm.compute()
        print("\n===== Calibration metrics ===== \n")
        for k, v in calib_metrics.items():
            print(f"{k.upper()}: {round(v, 4)}")
        print("\n=============================== \n")

    norms = np.concatenate(norms)
    avg_norm = np.mean(norms)
    print(f"Average norm: {avg_norm}")

    return losses.avg, acc1.avg, acc5.avg


# ----------------------------------
if __name__ == "__main__":
    args = get_arguments()

    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif args.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise "incorrect dtype in configuration -> " + args.dtype

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            print("Create missing output directory...")
            os.mkdir(args.output_dir)

    main(args)
