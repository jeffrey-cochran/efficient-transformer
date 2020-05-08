from torch.utils import data

from utils.data.Dataset_def import Dataset
from utils.data.CustomSampler_def import CustomSampler


class DataLoader:
    def __init__(
        self,
        sequences_per_img=5,
        batch_size=10,
        use_fc=True,
        use_att=True,
        use_box=0,
        norm_att_feat=0,
        norm_box_feat=0,
        input_json_file_name=None,
        input_label_h5_file_name=None,
    ):
        self.batch_size = batch_size
        self.dataset = Dataset(
            sequences_per_img=sequences_per_img,
            use_fc=use_fc,
            use_att=use_att,
            use_box=use_box,
            norm_att_feat=norm_att_feat,
            norm_box_feat=norm_box_feat,
            input_json_file_name=input_json_file_name,
            input_label_h5_file_name=input_label_h5_file_name,
        )

        # Initialize loaders and iters
        self.loaders, self.iters = {}, {}
        for split in ["train", "val", "test"]:
            if split == "train":
                sampler = CustomSampler(
                    self.dataset.split_ix[split], shuffle=True, wrap=True
                )
            else:
                sampler = CustomSampler(
                    self.dataset.split_ix[split], shuffle=False, wrap=False
                )
            self.loaders[split] = data.DataLoader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                pin_memory=True,
                num_workers=4,  # 4 is usually enough
                collate_fn=lambda x: self.dataset.collate_func(x, split),
                drop_last=False,
            )
            self.iters[split] = iter(self.loaders[split])

    def get_batch(self, split):
        try:
            data = next(self.iters[split])
        except StopIteration:
            self.iters[split] = iter(self.loaders[split])
            data = next(self.iters[split])
        return data

    def reset_iterator(self, split):
        self.loaders[split].sampler._reset_iter()
        self.iters[split] = iter(self.loaders[split])

    def get_vocab_size(self):
        return self.dataset.get_vocab_size()

    @property
    def vocab_size(self):
        return self.get_vocab_size()

    def get_vocab(self):
        return self.dataset.get_vocab()

    def get_seq_length(self):
        return self.dataset.get_seq_length()

    @property
    def seq_length(self):
        return self.get_seq_length()

    def state_dict(self):
        def get_prefetch_num(split):
            if self.loaders[split].num_workers > 0:
                return (
                    self.iters[split]._send_idx - self.iters[split]._rcvd_idx
                ) * self.batch_size
            else:
                return 0

        return {
            split: loader.sampler.state_dict(get_prefetch_num(split))
            for split, loader in self.loaders.items()
        }

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        for split in self.loaders.keys():
            self.loaders[split].sampler.load_state_dict(state_dict[split])
