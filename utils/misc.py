from copy import deepcopy
from torch.nn import ModuleList


def clones(module, N):
    "Produce N identical layers."
    return ModuleList([deepcopy(module) for _ in range(N)])
