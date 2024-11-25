import argparse
import os

import torch
from segmentation.models.classifier import SegformerClassifier
from segmentation.configs import cfg
from lib.geoopt import ManifoldParameter
from lib.geoopt.optim import RiemannianAdam, RiemannianSGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_checkpoint(model, optimizer, lr_scheduler, args):
    """Loads a checkpoint from file-system."""

    checkpoint = torch.load(args.load_checkpoint, map_location="cpu")

    model.load_state_dict(checkpoint["model"])

    if "optimizer" in checkpoint:
        if checkpoint["args"].optimizer == args.optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = args.lr

            if (lr_scheduler is not None) and ("lr_scheduler" in checkpoint):
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        else:
            print(
                "Warning: Could not load optimizer and lr-scheduler state_dict. Different optimizer in configuration ({}) and checkpoint ({}).".format(
                    args.optimizer, checkpoint["args"].optimizer
                )
            )

    epoch = 0
    if "epoch" in checkpoint:
        epoch = checkpoint["epoch"] + 1

    return model, optimizer, lr_scheduler, epoch


def load_model_checkpoint(model, checkpoint_path):
    """Loads a checkpoint from file-system."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    return model


def select_model(img_dim, num_classes, args):
    """Selects and sets up an available model and returns it."""

    enc_args = {
        "img_dim": img_dim,
        "embed_dim": args.embedding_dim,
        "num_classes": num_classes,
        "bias": args.encoder_manifold == "lorentz",
    }

    if args.encoder_manifold == "lorentz":
        enc_args["learn_k"] = args.learn_k
        enc_args["k"] = args.encoder_k

    dec_args = {
        "embed_dim": args.embedding_dim,
        "num_classes": num_classes,
        "k": args.decoder_k,
        "learn_k": args.learn_k,
        "type": "mlr",
        "clip_r": args.clip_features,
    }

    model = SegformerClassifier(
        enc_type=args.encoder_manifold,
        dec_type=args.decoder_manifold,
        num_classes=num_classes,
    )

    return model


def select_optimizer(model, args):
    """Selects and sets up an available optimizer and returns it."""

    model_parameters = get_param_groups(
        model, args.lr * args.lr_scheduler_gamma, args.weight_decay
    )

    if args.optimizer == "RiemannianAdam":
        optimizer = RiemannianAdam(
            model_parameters, lr=args.lr, weight_decay=args.weight_decay, stabilize=1
        )
    elif args.optimizer == "RiemannianSGD":
        optimizer = RiemannianSGD(
            model_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9,
            nesterov=True,
            stabilize=1,
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model_parameters, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model_parameters, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    else:
        raise "Optimizer not found. Wrong optimizer in configuration... -> " + args.model

    lr_scheduler = None
    if args.use_lr_scheduler:
        if args.lr_scheduler == "MultiStepLR":
            lr_scheduler = MultiStepLR(
                optimizer,
                milestones=args.lr_scheduler_milestones,
                gamma=args.lr_scheduler_gamma,
            )
        elif args.lr_scheduler == "CosineAnnealingLR":
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.num_epochs,  # eta_min=args.r * 0.01
            )
        else:
            raise "Learning rate scheduler not found. Wrong lr_scheduler in configuration... -> " + args.lr_scheduler

    return optimizer, lr_scheduler


def get_param_groups(model, lr_manifold, weight_decay_manifold):
    no_decay = ["scale"]
    k_params = ["manifold.k"]

    parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and not any(nd in n for nd in no_decay)
                and not isinstance(p, ManifoldParameter)
                and not any(nd in n for nd in k_params)
            ],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and isinstance(p, ManifoldParameter)
            ],
            "lr": lr_manifold,
            "weight_decay": weight_decay_manifold,
        },
        {  # k parameters
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in k_params)
            ],
            "weight_decay": 0,
            "lr": 1e-4,
        },
    ]

    return parameters


def select_dataset(args, validation_split=False):
    """Selects an available dataset and returns PyTorch dataloaders for training, validation and testing."""

    if args.dataset == "Cityscapes":
        root_dir = "segmentation/data/cityscapes/"

        w, h = 1280, 640

        train_transform = transforms.Compose(
            [
                transforms.Resize((h, w)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((h, w)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        target_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        train_set = datasets.Cityscapes(root=root_dir, split="train", mode="fine", target_type="semantic", transform=train_transform, target_transform=target_transform)
        val_set = datasets.Cityscapes(root=root_dir, split="val", mode="fine", target_type="semantic", transform=test_transform, target_transform=target_transform)
        test_set = datasets.Cityscapes(root=root_dir, split="val", mode="fine", target_type="semantic", transform=test_transform, target_transform=target_transform)

        img_dim = [3, h, w]
        num_classes = 19
    else:
        raise "Selected dataset '{}' not available.".format(args.dataset)

    # Dataloader
    batch_size = args.batch_size // len(args.gpus)
    batch_size_test = args.batch_size_test // len(args.gpus)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size_test,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
    )

    if validation_split:
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size_test,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
        )
    else:
        val_loader = test_loader

    return train_loader, val_loader, test_loader, img_dim, num_classes


def check_config(cfg):
    assert isinstance(cfg.exp_name, str) and cfg.exp_name != ""
    assert isinstance(cfg.output_dir, str) and cfg.output_dir != ""
    assert isinstance(cfg.gpus, list) and len(cfg.gpus) >= 1
    assert isinstance(cfg.dtype, str) and cfg.dtype in ["float32", "float64"]
    assert isinstance(cfg.seed, int) and cfg.seed > 0
    assert isinstance(cfg.load_checkpoint, (str, type(None)))
    assert isinstance(cfg.num_epochs, int) and cfg.num_epochs > 0
    assert isinstance(cfg.batch_size, int) and cfg.batch_size > 0
    assert isinstance(cfg.lr, float) and cfg.lr > 0
    assert isinstance(cfg.weight_decay, float) and cfg.weight_decay >= 0
    assert isinstance(cfg.optimizer, str) and cfg.optimizer in ["RiemannianAdam", "RiemannianSGD", "Adam", "SGD", "AdamW"]
    assert isinstance(cfg.use_lr_scheduler, bool)
    assert isinstance(cfg.lr_scheduler, str) and cfg.lr_scheduler in ["MultiStepLR", "CosineAnnealingLR"]
    assert isinstance(cfg.lr_scheduler_milestones, list) and all(isinstance(m, int) for m in cfg.lr_scheduler_milestones)
    assert isinstance(cfg.lr_scheduler_gamma, float) and cfg.lr_scheduler_gamma > 0
    assert isinstance(cfg.base_loss, str) and cfg.base_loss in ["cross_entropy", "focal"]
    assert isinstance(cfg.batch_size_test, int) and cfg.batch_size_test > 0
    assert isinstance(cfg.validation_split, bool)
    assert isinstance(cfg.num_layers, int) and cfg.num_layers in [18, 50]
    assert isinstance(cfg.embedding_dim, int) and cfg.embedding_dim > 0
    assert isinstance(cfg.encoder_manifold, str) and cfg.encoder_manifold in ["euclidean", "lorentz"]
    assert isinstance(cfg.decoder_manifold, str) and cfg.decoder_manifold in ["euclidean", "lorentz", "poincare"]
    assert isinstance(cfg.encoder_k, float) and cfg.encoder_k > 0
    assert isinstance(cfg.decoder_k, float) and cfg.decoder_k > 0
    assert isinstance(cfg.clip_features, float) and cfg.clip_features > 0
    assert isinstance(cfg.ral_initial_alpha, float) and cfg.ral_initial_alpha >= 0
    assert isinstance(cfg.ral_final_alpha, float) and cfg.ral_final_alpha >= 0
    assert isinstance(cfg.radius_conf_loss, float) and cfg.radius_conf_loss >= 0
    assert isinstance(cfg.radius_label_smoothing, bool)
    assert isinstance(cfg.dataset, str) and cfg.dataset in ["Cityscapes"]
    assert isinstance(cfg.wandb, bool)
    assert isinstance(cfg.notes, str)


def get_config():
    parser = argparse.ArgumentParser(
        description="Calibration of Hyperbolic Neural Networks"
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--proctitle",
        type=str,
        default="CPHNN",
        help="allow a process to change its title",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if args.opts is not None and args.opts != []:
        args.opts[-1] = args.opts[-1].strip("\r\n")

    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # add dataset config
    dataset_cfg_path = os.path.join("segmentation/configs/datasets", str(cfg.dataset).lower() + ".yaml")
    cfg.merge_from_file(dataset_cfg_path)

    # add model config
    model_cfg_path = os.path.join("segmentation/configs/models", str(cfg.model).lower() + ".yaml")
    cfg.merge_from_file(model_cfg_path)

    cfg.output_dir = os.path.join(cfg.output_dir, str(cfg.dataset).lower())
    print("Saving to {}".format(cfg.output_dir))
    check_config(cfg)
    cfg.freeze()

    return cfg
