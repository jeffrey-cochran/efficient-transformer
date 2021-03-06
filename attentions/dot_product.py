from torch import matmul
from math import sqrt
import torch.nn.functional as F


def dot_product_attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    print(f"QUERY: {query.shape}")
    print(f"KEY: {key.shape}")
    d_k = query.size(-1)
    scores = matmul(query, key.transpose(-2, -1)) / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return matmul(p_attn, value), p_attn
