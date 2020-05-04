#
# Import model components
from models.transformer.MultiHeadedAttention_def import MultiHeadedAttention
from models.transformer.PositionwiseFeedForward_def import PositionwiseFeedForward
from models.transformer.PositionalEncoding_def import PositionalEncoding
from models.transformer.TransformerEncoderDecoder_def import EncoderDecoder
from models.transformer.TransformerEncoder_def import TransformerEncoder
from models.transformer.TransformerEncoderLayer_def import TransformerEncoderLayer
from models.transformer.TransformerDecoder_def import TransformerDecoder
from models.transformer.TransformerDecoderLayer_def import TransformerDecoderLayer
from models.transformer.Embeddings_def import Embeddings
from models.transformer.Generator_def import Generator
#
# Other imports
from torch.nn import init, Sequential
from copy import deepcopy

def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        TransformerEncoder(TransformerEncoderLayer(d_model, c(attn), c(ff), dropout), N),
        TransformerDecoder(TransformerDecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Sequential(Embeddings(d_model, src_vocab), c(position)),
        Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            init.xavier_uniform(p)
    return model