from torch.nn import Module
from models.transformer.LayerNorm_def import LayerNorm
from models.transformer.utils import clones

class TransformerEncoder(Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(TransformerEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)