import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
import configargparse

def setup_environment():
    """Setup working directory and library paths."""
    working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
    os.chdir(working_dir)
    lib_path = os.path.join(working_dir)
    sys.path.append(lib_path)

setup_environment()

from utils.initialize import select_dataset, select_model, load_model_checkpoint
from lib.geoopt.manifolds.lorentz.math import dist0 as dist0_lorentz
from lib.geoopt.manifolds.stereographic.math import dist0 as dist0_poincare

def get_arguments():
    """Parses command-line options."""
    parser = configargparse.ArgumentParser(description='Image classification testing', add_help=True)
    
    # Configuration file
    parser.add_argument('-c', '--config_file', required=False, default=None, is_config_file=True, type=str, 
                        help="Path to config file.")
    
    # Modes
    parser.add_argument('--mode', default="test_confidence", type=str, choices=["test_confidence"],
                        help="Select the testing mode.")
    
    # Output settings
    parser.add_argument('--output_dir', default=None, type=str, 
                        help="Path for output files (relative to working directory).")
    
    # General settings
    parser.add_argument('--device', default="cuda:0", type=lambda s: [str(item) for item in s.replace(' ', '').split(',')],
                        help="List of devices split by comma (e.g. cuda:0,cuda:1), can also be a single device or 'cpu')")
    parser.add_argument('--dtype', default='float32', type=str, choices=["float32", "float64"], 
                        help="Set floating point precision.")
    parser.add_argument('--seed', default=1, type=int, 
                        help="Set seed for deterministic training.")
    parser.add_argument('--load_checkpoint', default='classification/output/best_L-ResNet18.pth', type=str, 
                        help="Path to model checkpoint.")
    
    # Testing parameters
    parser.add_argument('--batch_size', default=128, type=int, 
                        help="Training batch size.")
    parser.add_argument('--batch_size_test', default=128, type=int, 
                        help="Testing batch size.")
    
    # Model selection
    parser.add_argument('--num_layers', default=18, type=int, choices=[18, 50], 
                        help="Number of layers in ResNet.")
    parser.add_argument('--embedding_dim', default=512, type=int, 
                        help="Dimensionality of classification embedding space (could be expanded by ResNet).")
    parser.add_argument('--encoder_manifold', default='lorentz', type=str, choices=["lorentz", "euclidean"], 
                        help="Select conv model encoder manifold.")
    parser.add_argument('--decoder_manifold', default='lorentz', type=str, choices=["lorentz", "poincare"], 
                        help="Select conv model decoder manifold.")
    
    # Hyperbolic geometry settings
    parser.add_argument('--learn_k', action='store_true',
                        help="Set a learnable curvature of hyperbolic geometry.")
    parser.add_argument('--encoder_k', default=1.0, type=float, 
                        help="Initial curvature of hyperbolic geometry in backbone (geoopt.K=-1/K).")
    parser.add_argument('--decoder_k', default=1.0, type=float, 
                        help="Initial curvature of hyperbolic geometry in decoder (geoopt.K=-1/K).")
    parser.add_argument('--clip_features', default=1.0, type=float, 
                        help="Clipping parameter for hybrid HNNs proposed by Guo et al. (2022).")
    
    # Dataset settings
    parser.add_argument('--dataset', default='CIFAR-100', type=str, choices=["MNIST", "CIFAR-10", "CIFAR-100", "Tiny-ImageNet"], 
                        help="Select a dataset.")

    args, _ = parser.parse_known_args()

    return args

def evaluate(model, dataloader, device):
    """Evaluates model performance."""
    model.eval()
    model.to(device)

    predictions, probabilities, embeddings, labels = [], [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            labels.extend(y.numpy().tolist())

            x, y = x.to(device), y.to(device)

            logits = model(x)
            probabilities_batch = F.softmax(logits, dim=1)
            _, predicted = torch.max(probabilities_batch, 1)

            predictions.extend(predicted.cpu().tolist())
            probabilities.extend(probabilities_batch.cpu().tolist())

            embedding = model.module.embed(x)
            embeddings.extend(embedding.cpu().detach().numpy().tolist())
        
        embeddings = torch.tensor(embeddings, device=device)
        labels = torch.tensor(labels)

    return predictions, probabilities, embeddings, labels

def main(args):
    device = args.device[0]
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    print("Loading dataset...")
    _, val_loader, test_loader, img_dim, num_classes = select_dataset(args, validation_split=True)

    print("Selecting model...")
    model = select_model(img_dim, num_classes, args)
    model = model.to(device)
    model = load_model_checkpoint(model, args.load_checkpoint)
    model = DataParallel(model, device_ids=args.device)
    model.eval()

    print("Testing accuracy of model...")
    predictions, probabilities, embeddings, labels = evaluate(model, test_loader, device)

    dist0 = dist0_lorentz if args.decoder_manifold == 'lorentz' else dist0_poincare

    results = []

    for prediction, probability, label, embedding in zip(predictions, probabilities, labels, embeddings):

        result = {
            'prediction': prediction,                                                       # Predicted Value
            'confidence': max(probability),                                                 # Probability of the predicted Value
            'label': label.item(),                                                          # True Value of the instance
            'hyper_radius': dist0(embedding, k=torch.tensor(1.0)).detach().cpu().item()     # Hyperbolic radius of the embedding
        }
        results.append(result)

    print("Finished!")
    return results

if __name__ == '__main__':
    args = get_arguments()

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    results = main(args)

    # Save the results
    import pandas as pd

    df = pd.DataFrame(results)
    df['norm_radius'] = (                                                                   # Normalized Hyperbolic radius
        df['hyper_radius'] - df['hyper_radius'].min()) / (df['hyper_radius'].max() - df['hyper_radius'].min()) 

    output_path = os.path.join(args.output_dir, 'results.csv') if args.output_dir else 'classification/results.csv'
    df.to_csv(output_path, index=False)
