from torch.nn import Module, Dropout
from torch import zeros, arange, exp, sin, cos
from math import log
from torch.autograd import Variable

class PositionalEncoding(Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = zeros(max_len, d_model)
        position = arange(0, max_len).unsqueeze(1)
        div_term = exp(arange(0, d_model, 2) *
                             -(log(10000.0) / d_model))
        pe[:, 0::2] = sin(position * div_term)
        pe[:, 1::2] = cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)