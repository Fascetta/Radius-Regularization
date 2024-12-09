import torch
import torch.nn as nn
from models.segformer_head import SegformerDecodeHead
from transformers import SegformerModel


class SegformerClassifier(nn.Module):
    """Classifier based on ResNet encoder."""

    def __init__(
        self,
        enc_type: str = "euclidean",
        dec_type: str = "poincare",
        num_classes: int = 19,
        model_size: str = "b5",
    ):
        super(SegformerClassifier, self).__init__()

        self.enc_type = enc_type
        self.dec_type = dec_type
        self.model_size = model_size

        self.encoder = SegformerModel.from_pretrained(f"nvidia/mit-{self.model_size}")
        self.encoder_config = self.encoder.config

        self.dec_manifold = None
        if dec_type == "euclidean":
            hyperbolic = False
        elif dec_type == "poincare":
            hyperbolic = True
        else:
            raise RuntimeError(f"Decoder manifold {dec_type} not available...")

        self.decoder = SegformerDecodeHead(
            self.encoder_config, num_classes=num_classes, hyper=hyperbolic, hfr=False
        )

    def embed(self, x):
        return_dict = self.encoder_config.return_dict
        x = self.encoder(x, output_hidden_states=True, return_dict=return_dict)
        embeds = x.hidden_states if return_dict else x[1]
        return embeds

    def forward(self, x, size=None):
        x = self.embed(x)
        logits, radii = self.decoder(x, size=size)
        return logits, radii
