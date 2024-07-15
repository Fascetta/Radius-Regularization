# -----------------------------------------------------
# Change working directory to parent HyperbolicCV/code
import os
import sys

working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)
# -----------------------------------------------------

import torch
from torch.nn import DataParallel

import configargparse
from tqdm import tqdm

import random
import numpy as np

from classification.utils.initialize import select_dataset, select_model, select_optimizer, load_checkpoint
from lib.utils.utils import AverageMeter, accuracy
from classification.utils.calibration_metrics import CalibrationMetrics
from torchmetrics.classification import MulticlassCalibrationError
from scipy.optimize import minimize


def getArguments():
    """ Parses command-line options. """
    parser = configargparse.ArgumentParser(description='Image classification training', add_help=True)

    parser.add_argument('-c', '--config_file', required=False, default=None, is_config_file=True, type=str,
                        help="Path to config file.")

    # Output settings
    parser.add_argument('--exp_name', default="test", type=str,
                        help="Name of the experiment.")
    parser.add_argument('--output_dir', default=None, type=str,
                        help="Path for output files (relative to working directory).")

    # General settings
    parser.add_argument('--device', default="cuda:0",
                        type=lambda s: [str(item) for item in s.replace(' ', '').split(',')],
                        help="List of devices split by comma (e.g. cuda:0,cuda:1), can also be a single device or 'cpu')")
    parser.add_argument('--dtype', default='float32', type=str, choices=["float32", "float64"],
                        help="Set floating point precision.")
    parser.add_argument('--seed', default=1, type=int,
                        help="Set seed for deterministic training.")
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help="Path to model checkpoint (weights, optimizer, epoch).")
    parser.add_argument('--compile', action='store_true',
                        help="Compile model for faster inference (requires PyTorch 2).")

    # General training parameters
    parser.add_argument('--num_epochs', default=200, type=int,
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', default=128, type=int,
                        help="Training batch size.")
    parser.add_argument('--lr', default=1e-1, type=float,
                        help="Training learning rate.")
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help="Weight decay (L2 regularization)")
    parser.add_argument('--optimizer', default="RiemannianSGD", type=str,
                        choices=["RiemannianAdam", "RiemannianSGD", "Adam", "SGD"],
                        help="Optimizer for training.")
    parser.add_argument('--use_lr_scheduler', action='store_true',
                        help="If learning rate should be reduced after step epochs using a LR scheduler.")
    parser.add_argument('--lr_scheduler_milestones', default=[60, 120, 160], type=int, nargs="+",
                        help="Milestones of LR scheduler.")
    parser.add_argument('--lr_scheduler_gamma', default=0.2, type=float,
                        help="Gamma parameter of LR scheduler.")

    # General validation/testing hyperparameters
    parser.add_argument('--batch_size_test', default=128, type=int,
                        help="Validation/Testing batch size.")
    parser.add_argument('--validation_split', action='store_true',
                        help="Use validation split of training data.")

    # Model selection
    parser.add_argument('--num_layers', default=18, type=int, choices=[18, 50],
                        help="Number of layers in ResNet.")
    parser.add_argument('--embedding_dim', default=512, type=int,
                        help="Dimensionality of classification embedding space (could be expanded by ResNet)")
    parser.add_argument('--encoder_manifold', default='lorentz', type=str, choices=["euclidean", "lorentz"],
                        help="Select conv model encoder manifold.")
    parser.add_argument('--decoder_manifold', default='lorentz', type=str, choices=["euclidean", "lorentz", "poincare"],
                        help="Select conv model decoder manifold.")

    # Hyperbolic geometry settings
    parser.add_argument('--learn_k', action='store_true',
                        help="Set a learnable curvature of hyperbolic geometry.")
    parser.add_argument('--encoder_k', default=1.0, type=float,
                        help="Initial curvature of hyperbolic geometry in backbone (geoopt.K=-1/K).")
    parser.add_argument('--decoder_k', default=1.0, type=float,
                        help="Initial curvature of hyperbolic geometry in decoder (geoopt.K=-1/K).")
    parser.add_argument('--clip_features', default=1.0, type=float,
                        help="Clipping parameter for hybrid HNNs proposed by Guo et al. (2022)")
    parser.add_argument('--radius_loss', default=0.0, type=float,
                        help="Use radius focal loss together with cross-entropy loss.")

    # Dataset settings
    parser.add_argument('--dataset', default='CIFAR-100', type=str,
                        choices=["MNIST", "CIFAR-10", "CIFAR-100", "Tiny-ImageNet"],
                        help="Select a dataset.")

    args = parser.parse_args()

    return args


def get_radius_tau(model, val_loader, criterion, device):

    def scaled_cross_entropy_loss(embeds, labels, tau):
        embeds, labels = embeds.to(device), labels.to(device)
        tau = torch.tensor(tau, dtype=torch.float32, device=device)
        scaled_embeds = embeds / tau
        logits = model.module.decoder(scaled_embeds)
        loss = criterion(logits, labels)
        return loss.item()
    
    embeddings_list, labels_list = [], []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            embeds = model.module.embed(x)
            embeddings_list.append(embeds.cpu())
            labels_list.append(y.cpu())

    embeds = torch.cat(embeddings_list)
    labels = torch.cat(labels_list)


    res = minimize(lambda tau: scaled_cross_entropy_loss(embeds, labels, tau), 1,  bounds=[(0.05, 5.0)], options={'eps':0.01})

    optimal_tau = res.x[0]

    return optimal_tau


def get_confidence_tau(logits, labels, criterion):

    def scaled_cross_entropy_loss(logits, labels, tau):
        scaled_logits = logits / tau
        loss = criterion(scaled_logits, labels)
        return loss.item()

    initial_tau = 1.0

    res = minimize(lambda tau: scaled_cross_entropy_loss(logits, labels, tau), initial_tau, method='L-BFGS-B', bounds=[(0.05, 3.0)])

    optimal_tau = res.x[0]

    return optimal_tau


def compute_estimated_confidence(logits, labels):
    """Computes the estimated confidence for each class."""
    n_classes = logits.shape[-1]
    confidence = torch.zeros(n_classes, device=logits.device)
    preds = logits.argmax(dim=-1)

    # compute total counts and correct counts for each class
    total_counts = torch.bincount(labels, minlength=n_classes).float()
    correct = (preds == labels).float()
    correct_counts = torch.zeros(n_classes, device=logits.device)
    correct_counts.index_add_(0, labels, correct.float())

    # compute confidence for each class
    confidence = correct_counts / total_counts
    confidence[total_counts == 0] = 0.0
    return confidence


def radius_confidence_loss(logits, labels, radii):
    confidence = compute_estimated_confidence(logits, labels)
    true_class_confidence = confidence[labels]
    loss = torch.nn.functional.mse_loss(radii, true_class_confidence)
    return loss


def main(args):
    device = args.device[0]
    # torch.cuda.set_device(device)     deprecated
    torch.cuda.empty_cache()

    print("Running experiment: " + args.exp_name)

    print("Arguments:")
    print(args)

    print(f"Loading dataset with validation_split = {args.validation_split}...")
    train_loader, val_loader, test_loader, img_dim, num_classes = select_dataset(args, validation_split=args.validation_split)

    print("Creating model...")
    model = select_model(img_dim, num_classes, args)
    model = model.to(device)
    print('-> Number of model params: {} (trainable: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    print("Creating optimizer...")
    optimizer, lr_scheduler = select_optimizer(model, args)
    criterion = torch.nn.CrossEntropyLoss()

    use_radius_loss = args.radius_loss > 0.0
    if use_radius_loss:
        print(f"Using radius loss with alpha={args.radius_loss}")
        alpha = args.radius_loss
        # radius_loss = lambda x: torch.mean(x ** 2) * alpha
        radius_loss = radius_confidence_loss

    start_epoch = 0
    if args.load_checkpoint is not None:
        print("Loading model checkpoint from {}".format(args.load_checkpoint))
        model, optimizer, lr_scheduler, start_epoch = load_checkpoint(model, optimizer, lr_scheduler, args)

    model = DataParallel(model, device_ids=args.device)

    if args.compile:
        model = torch.compile(model)

    print("Training...")
    global_step = start_epoch * len(train_loader)

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, args.num_epochs):
        model.train()

        losses = AverageMeter("Loss", ":.4e")
        radius_losses = AverageMeter("RadiusLoss", ":.4e")
        acc1 = AverageMeter("Acc@1", ":6.2f")
        acc5 = AverageMeter("Acc@5", ":6.2f")

        for i, (x, y) in tqdm(enumerate(train_loader)):
            # ------- Start iteration -------
            x = x.to(device)
            y = y.to(device)

            if use_radius_loss:
                embeds = model.module.embed(x)
                logits = model.module.decoder(embeds)
                radii = torch.norm(embeds, dim=-1, p=2)
                rl = radius_loss(logits, y, radii)
                ce_loss = criterion(logits, y)
                loss = ce_loss + rl * alpha
            else:
                logits = model(x)
                loss = criterion(logits, y) # Compute loss

            optimizer.zero_grad() # Reset gradients tensoes
            loss.backward() # Back-Propagation

            optimizer.step() # Update optimazer

            with torch.no_grad():
                top1, top5 = accuracy(logits, y, topk=(1, 5))
                losses.update(loss.item())
                if use_radius_loss:
                    radius_losses.update(rl.item())
                acc1.update(top1.item())
                acc5.update(top5.item())

            global_step += 1
            # ------- End iteration -------

        # ------- Start validation and logging -------
        with torch.no_grad():
            if lr_scheduler is not None:
                if (epoch + 1) == args.lr_scheduler_milestones[0]:  # skip the first drop for some Parameters
                    optimizer.param_groups[1]['lr'] *= (1 / args.lr_scheduler_gamma) # Manifold params
                    print("Skipped lr drop for manifold parameters")

                lr_scheduler.step()

            loss_val, acc1_val, acc5_val = evaluate(model, val_loader, criterion, device)

            if use_radius_loss:
                print(
                    "Epoch {}/{}: Loss={:.4f}, RadiusLoss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}, Validation: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
                        epoch + 1, args.num_epochs, losses.avg, radius_losses.avg, acc1.avg, acc5.avg, loss_val, acc1_val, acc5_val))
            else:
                print(
                    "Epoch {}/{}: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}, Validation: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
                        epoch + 1, args.num_epochs, losses.avg, acc1.avg, acc5.avg, loss_val, acc1_val, acc5_val))

            # Testing for best model
            if acc1_val > best_acc:
                best_acc = acc1_val
                best_epoch = epoch + 1
                if args.output_dir is not None:
                    save_path = args.output_dir + "/best_" + args.exp_name + ".pth"
                    torch.save({
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                        'epoch': epoch,
                        'args': args,
                    }, save_path)
        # ------- End validation and logging -------

    print("-----------------\nTraining finished\n-----------------")
    print("Best epoch = {}, with Acc@1={:.4f}".format(best_epoch, best_acc))

    if args.output_dir is not None:
        save_path = args.output_dir + "/final_" + args.exp_name + ".pth"
        torch.save({
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'epoch': epoch,
            'args': args,
        }, save_path)
        print("Model saved to " + save_path)
    else:
        print("Model not saved.")

    print("Testing final model...")
    loss_test, acc1_test, acc5_test = evaluate(model, test_loader, criterion, device)

    print("Results: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
        loss_test, acc1_test, acc5_test))

    print("Testing best model...")
    if args.output_dir is not None:
        print("Loading best model...")
        save_path = args.output_dir + "/best_" + args.exp_name + ".pth"
        checkpoint = torch.load(save_path, map_location=device)
        model.module.load_state_dict(checkpoint['model'], strict=True)

        loss_test, acc1_test, acc5_test = evaluate(model, test_loader, criterion, device)

        print("Results: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
            loss_test, acc1_test, acc5_test))
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
        cm = CalibrationMetrics(n_classes = 100)

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
if __name__ == '__main__':
    args = getArguments()

    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif args.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise "Wrong dtype in configuration -> " + args.dtype

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            print("Create missing output directory...")
            os.mkdir(args.output_dir)

    main(args)
    