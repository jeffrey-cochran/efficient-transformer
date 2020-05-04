from torch.nn import Module, Embedding
from math import sqrt

class Embeddings(Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * sqrt(self.d_model)