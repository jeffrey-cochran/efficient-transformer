from copy import deepcopy
from torch.nn import ModuleList
from torch import is_tensor
from six import PY3
from six.moves import cPickle
from os.path import join, isdir
from os import makedirs
from torch import save as torch_save


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


def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: pickled object
    f: file-like object
    """
    if PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)


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


def save_checkpoint(
    model,
    infos,
    optimizer,
    checkpoint_dir=None,
    job_id=None,
    histories=None,
    append="",
):
    #
    # Modify appendage
    if len(append) > 0:
        append = "-" + append
    #
    # if checkpoint_dir doesn't exist, create it
    if not isdir(checkpoint_dir):
        makedirs(checkpoint_dir)
    #
    # Set file names
    checkpoint_path = join(checkpoint_dir, f"model{append}.pth")
    optimizer_path = join(checkpoint_dir, f"optimizer{append}.pth")
    infos_path = join(checkpoint_dir, f"infos_{job_id}{append}.pkl")
    histories_path = join(checkpoint_dir, f"histories_{job_id}{append}.pkl")

    #
    # Save checkpoint data
    print(f"Saving checkpoint to {checkpoint_path}")
    torch_save(model.state_dict(), checkpoint_path)

    #
    # Save optimizer data
    torch_save(optimizer.state_dict(), optimizer_path)

    #
    # Save infos data
    with open(infos_path, "wb") as f:
        pickle_dump(infos, f)

    #
    # Save histories data
    if histories is not None:
        with open(histories_path, "wb") as f:
            pickle_dump(histories, f)
