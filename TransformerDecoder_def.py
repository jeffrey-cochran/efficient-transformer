from torch.nn import Module
from models.transformer.utils import clones
from models.transformer.LayerNorm_def import LayerNorm

class TransformerDecoder(Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(TransformerDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)