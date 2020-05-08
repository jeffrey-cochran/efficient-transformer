from numpy import array as np_array
from numpy import load as np_load
from torch import load as torch_load
from os.path import isdir, join
from lmdb import open as lmdb_open
from h5py import File as H5File
from six import BytesIO


class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.
    """

    def __init__(self, db_path, ext):
        self.db_path = db_path
        self.ext = ext
        if self.ext == ".npy":
            self.loader = lambda x: np_load(x)
        else:
            self.loader = lambda x: np_load(x)["feat"]
        if db_path.endswith(".lmdb"):
            self.db_type = "lmdb"
            self.env = lmdb_open(
                db_path,
                subdir=isdir(db_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        elif db_path.endswith(".pth"):  # Assume a key,value dictionary
            self.db_type = "pth"
            self.feat_file = torch_load(db_path)
            self.loader = lambda x: x
            print("HybridLoader: ext is ignored")
        elif db_path.endswith("h5"):
            self.db_type = "h5"
            self.loader = lambda x: np_array(x).astype("float32")
        else:
            self.db_type = "dir"

    def get(self, key):

        if self.db_type == "lmdb":
            env = self.env
            with env.begin(write=False) as txn:
                byteflow = txn.get(key.encode())
            f_input = BytesIO(byteflow)
        elif self.db_type == "pth":
            f_input = self.feat_file[key]
        elif self.db_type == "h5":
            f_input = H5File(self.db_path, "r")[key]
        else:
            f_input = join(self.db_path, key + self.ext)

        # load image
        feat = self.loader(f_input)

        return feat
