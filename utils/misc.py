from copy import deepcopy
from torch.nn import ModuleList
from torch import is_tensor
from six import PY3
from six.moves import cPickle


def clones(module, N):
    "Produce N identical layers."
    return ModuleList([deepcopy(module) for _ in range(N)])


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def pickle_load(f):
    """ Load a pickle.
    Parameters
    ----------
    f: file-like object
    """
    if PY3:
        return cPickle.load(f, encoding="latin-1")
    else:
        return cPickle.load(f)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group["lr"]
