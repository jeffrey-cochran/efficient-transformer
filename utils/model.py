#
# Import model components
from model.MultiHeadedAttention_def import MultiHeadedAttention
from model.PositionwiseFeedForward_def import PositionwiseFeedForward
from model.PositionalEncoding_def import PositionalEncoding
from model.EncoderDecoder_def import EncoderDecoder
from model.Encoder_def import Encoder
from model.EncoderLayer_def import EncoderLayer
from model.Decoder_def import Decoder
from model.DecoderLayer_def import DecoderLayer
from model.Embeddings_def import Embeddings
from model.Generator_def import Generator

#
# Other imports
from torch.nn import init, Sequential
from copy import deepcopy


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = deepcopy
    attn = MultiHeadedAttention(h, d_model)  # Why add dropout?
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        lambda x: x,  # Embed before passing...Sequential(Embeddings(d_model, src_vocab), c(position)),
        Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    #
    # Init parameters randomly
    for p in model.parameters():
        if p.dim() > 1:
            init.xavier_uniform(p)
    return model
