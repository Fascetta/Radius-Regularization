import os

from yacs.config import CfgNode as CN

_C = CN()

# Output settings
_C.exp_name = "EP-ResNet18_RAL"
_C.notes = ""

_C.output_dir = "code/segmentation/output"

# General settings
_C.gpus = [1]
_C.dtype = "float32"
_C.seed = 1
_C.debug = False

# Test
_C.checkpoint_path = None
_C.mode = None
_C.calibration = None

# General training hyperparameters
_C.num_epochs = 200
_C.train_batch_size = 128
_C.lr = 1e-1
_C.weight_decay = 5e-4
_C.optimizer = "RiemannianSGD"
_C.use_lr_scheduler = True
_C.lr_scheduler = "MultiStepLR"             # CosineAnnealingLR or MultiStepLR
_C.lr_scheduler_milestones = [60, 120, 160]
_C.lr_scheduler_gamma = 0.2
_C.base_loss = "cross_entropy"               # cross_entropy or focal_loss
_C.ral_initial_alpha = 0.1
_C.ral_final_alpha = 0.1
_C.radius_conf_loss = 0.
_C.radius_label_smoothing = False

# General validation/testing hyperparameters
_C.test_batch_size = 32
_C.validation_split = False

# Model selection
_C.model_size = "b2"
_C.num_layers = 18
_C.embedding_dim = 512
_C.encoder_manifold = "euclidean"
_C.decoder_manifold = "poincare"

# Manifold settings
_C.learn_k = False
_C.encoder_k = 1.0
_C.decoder_k = 1.0
_C.clip_features = 1.0

# Model settings
_C.model = "euclidean-poincare"

# Dataset settings
_C.dataset = "CIFAR-100"         # CIFAR-10 or Tiny-ImageNet

# Logging
_C.wandb = False