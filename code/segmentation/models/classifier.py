import torch
import torch.nn as nn

from transformers import SegformerModel
from segmentation.models.segformer import SegformerDecodeHead


class SegformerClassifier(nn.Module):
    """ Classifier based on ResNet encoder.
    """
    def __init__(self, 
            enc_type:str="euclidean", 
            dec_type:str="poincare",
            num_classes:int=19,
        ):
        super(SegformerClassifier, self).__init__()

        self.enc_type = enc_type
        self.dec_type = dec_type

        self.encoder = SegformerModel.from_pretrained(f"nvidia/mit-b2")
        self.encoder_config = self.encoder.config

        self.dec_manifold = None
        if dec_type == "euclidean":
            hyperbolic = False
        elif dec_type == "poincare":
            hyperbolic = True
        else:
            raise RuntimeError(f"Decoder manifold {dec_type} not available...")

        self.decoder = SegformerDecodeHead(self.encoder_config, num_classes=num_classes, hyper=hyperbolic, hfr=False)
    
    def embed(self, x):
        return_dict = self.encoder_config.return_dict
        x = self.encoder(x, output_hidden_states=True, return_dict=return_dict)
        embeds = x.hidden_states if return_dict else x[1]
        return embeds

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.embed(x)
        logits, hidden_states = self.decoder(x, size=input_size)
        return logits, hidden_states

        

