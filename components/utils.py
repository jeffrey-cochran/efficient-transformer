#
# Other imports
from copy import deepcopy
from torch.nn import ModuleList
from torch import from_numpy
from numpy import triu, ones


def clones(module, N):
    "Produce N identical layers."
    return ModuleList([deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = triu(ones(attn_shape), k=1).astype("uint8")
    return from_numpy(subsequent_mask) == 0
