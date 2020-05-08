#
# External packages
from json import load as load_json
from h5py import File as H5File
from numpy import zeros as np_zeros
from numpy import stack as np_stack
from numpy import vstack as np_vstack
from numpy import array as np_array
from numpy import ndarray as np_ndarray
from numpy import hsplit as np_hsplit, hstack as np_hstack
from numpy.linalg import norm as np_norm
from torch import from_numpy
from torch.utils import data
from random import randint

#
# Internal modules
from utils.constants import input_fc_dir, input_att_dir, input_box_dir
from utils.data.HybridLoader_def import HybridLoader


class Dataset(data.Dataset):
    def __init__(
        self,
        sequences_per_img=5,
        use_fc=True,
        use_att=True,
        use_box=0,
        norm_att_feat=0,
        norm_box_feat=0,
        input_json_file_name=None,
        input_label_h5_file_name=None,
    ):
        #
        self.necessary_num_img_captions = sequences_per_img

        #
        # feature related options
        self.use_fc = use_fc
        self.use_att = use_att
        self.use_box = use_box
        self.norm_att_feat = norm_att_feat
        self.norm_box_feat = norm_box_feat

        #
        # load the json file which contains additional information about the dataset
        print("DataLoader loading json file: ", input_json_file_name)
        with open(input_json_file_name) as input_json_file:
            self.info = load_json(input_json_file)

        if "ix_to_word" in self.info:
            self.ix_to_word = self.info["ix_to_word"]
            self.vocab_size = len(self.ix_to_word)
            print("vocab size is ", self.vocab_size)

        #
        # open the hdf5 file
        # NOTE: Setting input_label_h5_file_name to None is used when only doing generation.
        #       For example, when you need to test on coco test set.
        print(
            "DataLoader loading h5 file: ",
            input_fc_dir,
            input_att_dir,
            input_box_dir,
            input_label_h5_file_name,
        )

        if input_label_h5_file_name is not None:
            #
            # Open file
            self.h5_label_file = H5File(input_label_h5_file_name, "r", driver="core")

            #
            # Load in the sequence data
            seq_size = self.h5_label_file["labels"].shape
            self.label = self.h5_label_file["labels"][:]
            self.max_seq_length = seq_size[1]
            print("max sequence length in data is", self.max_seq_length)

            #
            # Load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.h5_label_file["label_start_ix"][:]
            self.label_end_ix = self.h5_label_file["label_end_ix"][:]
        else:
            self.max_seq_length = 1

        #
        # Feature loaders
        self.fc_loader = HybridLoader(input_fc_dir, ".npy")
        self.att_loader = HybridLoader(input_att_dir, ".npz")
        self.box_loader = HybridLoader(input_box_dir, ".npy")

        #
        # Image info
        self.num_images = len(self.info["images"])
        print("read %d image features" % (self.num_images))

        #
        # Split data into train, validation, and testing sets
        self.split_ix = {"train": [], "val": [], "test": []}
        for ix in range(self.num_images):
            #
            current_img = self.info["images"][ix]
            if not ("split" in current_img):
                self.split_ix["train"].append(ix)
                self.split_ix["val"].append(ix)
                self.split_ix["test"].append(ix)
            elif current_img["split"] == "train":
                self.split_ix["train"].append(ix)
            elif current_img["split"] == "val":
                self.split_ix["val"].append(ix)
            elif current_img["split"] == "test":
                self.split_ix["test"].append(ix)
            else:
                self.split_ix["train"].append(ix)

        print("assigned %d images to split train" % len(self.split_ix["train"]))
        print("assigned %d images to split val" % len(self.split_ix["val"]))
        print("assigned %d images to split test" % len(self.split_ix["test"]))

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.max_seq_length

    def get_captions(self, ix, necessary_num_img_captions):
        #
        # Fetch the sequence labels
        # NOTE: 1-indexed, not 0-indexed
        first_caption_idx = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
        last_caption_idx = self.label_end_ix[ix] - 1
        num_img_captions = last_caption_idx - first_caption_idx + 1
        assert num_img_captions > 0, f"Image {ix} has no caption. Aborting!"

        #
        # If we require more captions per image for training
        # than are available, we sample with replacement.
        if num_img_captions < necessary_num_img_captions:
            #
            seq = np_zeros(
                [necessary_num_img_captions, self.max_seq_length], dtype="int"
            )
            for q in range(necessary_num_img_captions):
                ixl = randint(first_caption_idx, last_caption_idx)
                seq[q, :] = self.label[ixl, : self.max_seq_length]
        else:
            #
            # Unnecessary to choose the captions sequentially. Come back to this later...
            ixl = randint(
                first_caption_idx, last_caption_idx - necessary_num_img_captions + 1
            )
            seq = self.label[
                ixl : ixl + necessary_num_img_captions, : self.max_seq_length
            ]

        return seq

    def collate_func(self, batch, split):

        necessary_num_img_captions = self.necessary_num_img_captions

        fc_batch = []
        att_batch = []
        label_batch = []

        wrapped = False

        infos = []
        gts = []

        for sample in batch:
            # fetch image
            tmp_fc, tmp_att, tmp_seq, ix, it_pos_now, tmp_wrapped = sample
            if tmp_wrapped:
                wrapped = True

            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)

            tmp_label = np_zeros(
                [necessary_num_img_captions, self.max_seq_length + 2], dtype="int"
            )
            if hasattr(self, "h5_label_file"):
                # if there is ground truth
                tmp_label[:, 1 : self.max_seq_length + 1] = tmp_seq
            label_batch.append(tmp_label)

            # Used for reward evaluation
            if hasattr(self, "h5_label_file"):
                # if there is ground truth
                gts.append(
                    self.label[self.label_start_ix[ix] - 1 : self.label_end_ix[ix]]
                )
            else:
                gts.append([])

            # record associated info as well
            info_dict = {}
            info_dict["ix"] = ix
            info_dict["id"] = self.info["images"][ix]["id"]
            info_dict["file_path"] = self.info["images"][ix].get("file_path", "")
            infos.append(info_dict)

        # #sort by att_feat length
        # fc_batch, att_batch, label_batch, gts, infos = \
        #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
        fc_batch, att_batch, label_batch, gts, infos = zip(
            *sorted(
                zip(fc_batch, att_batch, label_batch, gts, infos),
                key=lambda x: 0,
                reverse=True,
            )
        )
        data = {}
        data["fc_feats"] = np_stack(fc_batch)
        # merge att_feats
        max_att_len = max([_.shape[0] for _ in att_batch])
        data["att_feats"] = np_zeros(
            [len(att_batch), max_att_len, att_batch[0].shape[1]], dtype="float32"
        )
        for i in range(len(att_batch)):
            data["att_feats"][i, : att_batch[i].shape[0]] = att_batch[i]
        data["att_masks"] = np_zeros(data["att_feats"].shape[:2], dtype="float32")
        for i in range(len(att_batch)):
            data["att_masks"][i, : att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if data["att_masks"].sum() == data["att_masks"].size:
            data["att_masks"] = None

        data["labels"] = np_vstack(label_batch)
        # generate mask
        nonzeros = np_vstack(list(map(lambda x: (x != 0).sum() + 2, data["labels"])))
        mask_batch = np_zeros(
            [data["labels"].shape[0], self.max_seq_length + 2], dtype="float32"
        )
        for ix, row in enumerate(mask_batch):
            row[: nonzeros[ix]] = 1
        data["masks"] = mask_batch
        data["labels"] = data["labels"].reshape(
            len(batch), necessary_num_img_captions, -1
        )
        data["masks"] = data["masks"].reshape(
            len(batch), necessary_num_img_captions, -1
        )

        data["gts"] = gts  # all ground truth captions of each images
        data["bounds"] = {
            "it_pos_now": it_pos_now,  # the it_pos_now of the last sample
            "it_max": len(self.split_ix[split]),
            "wrapped": wrapped,
        }
        data["infos"] = infos

        data = {
            k: from_numpy(v) if type(v) is np_ndarray else v for k, v in data.items()
        }  # Turn all ndarray to torch tensor

        return data

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix, it_pos_now, wrapped = index  # self.split_ix[index]
        if self.use_att:
            att_feat = self.att_loader.get(str(self.info["images"][ix]["id"]))
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if self.norm_att_feat:
                att_feat = att_feat / np_norm(att_feat, 2, 1, keepdims=True)
            if self.use_box:
                box_feat = self.box_loader.get(str(self.info["images"][ix]["id"]))
                # devided by image width and height
                x1, y1, x2, y2 = np_hsplit(box_feat, 4)
                h, w = (
                    self.info["images"][ix]["height"],
                    self.info["images"][ix]["width"],
                )
                box_feat = np_hstack(
                    (x1 / w, y1 / h, x2 / w, y2 / h, (x2 - x1) * (y2 - y1) / (w * h))
                )  # question? x2-x1+1??
                if self.norm_box_feat:
                    box_feat = box_feat / np_norm(box_feat, 2, 1, keepdims=True)
                att_feat = np_hstack([att_feat, box_feat])
                # sort the features by the size of boxes
                att_feat = np_stack(sorted(att_feat, key=lambda x: x[-1], reverse=True))
        else:
            att_feat = np_zeros((0, 0), dtype="float32")
        if self.use_fc:
            try:
                fc_feat = self.fc_loader.get(str(self.info["images"][ix]["id"]))
            except:
                # Use average of attention when there is no fc provided (For bottomup feature)
                fc_feat = att_feat.mean(0)
        else:
            fc_feat = np_zeros((0), dtype="float32")
        if hasattr(self, "h5_label_file"):
            seq = self.get_captions(ix, self.necessary_num_img_captions)
        else:
            seq = None
        return (fc_feat, att_feat, seq, ix, it_pos_now, wrapped)

    def __len__(self):
        return len(self.info["images"])
