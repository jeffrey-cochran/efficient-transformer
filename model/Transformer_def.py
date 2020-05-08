from torch.nn import Sequential, Linear, ReLU, Dropout, Module
from torch import long as torch_long
from torch import cat as torch_cat
from torch import sort as torch_sort
from torch import arange as torch_arange
from utils.masks import subsequent_mask
from utils.misc import repeat_tensors
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


class Transformer(Module):
    #
    #
    def __init__(
        self,
        tgt_vocab_size,
        N=6,
        d_model=512,
        d_ff=2048,
        h=8,
        dropout=0.1,
        src_vocab_size=None,
    ):
        #
        # Initialize super model
        super(Transformer, self).__init__()

        #
        # Set hyperparams
        self.N_enc = N
        self.N_dec = N  # assume that equal number of layers on either side
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.dropout = dropout

        #
        # Function for data embedding
        self.embed = lambda x: x
        self.fc_embed = lambda x: x
        self.att_embed = Sequential(
            Linear(self.att_feat_size, self.d_model), ReLU(), Dropout(dropout)
        )

        #
        # Note that vocab size is increased by one for END token
        self.model = self.make_model(
            0,
            tgt_vocab_size + 1,
            N_enc=self.N_enc,
            N_dec=self.N_dec,
            d_model=self.d_model,
            d_ff=self.d_ff,
            h=self.h,
            dropout=self.dropout,
        )

    def logit(self, x):  # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    def prepare_feature_for_generation(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self.prepare_feature_for_forward_prop(
            att_feats, att_masks
        )
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :0], att_feats[..., :0], memory, att_masks

    def prepare_feature_for_forward_prop(self, att_feats, att_masks=None, seq=None):

        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = self.pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch_long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = seq.data > 0  # Why mask negative?
            seq_mask[:, 0] = 1  # bos

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks = repeat_tensors(
                    seq_per_img, [att_feats, att_masks]
                )
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def forward(self, fc_feats, att_feats, seq, att_masks=None):
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(
            att_feats, att_masks, seq
        )

        out = self.model(att_feats, seq, att_masks, seq_mask)

        outputs = self.model.generator(out)
        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch_cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(
            memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device)
        )
        return out[:, -1], [ys.unsqueeze(0)]

    def clip_att(self, att_feats, att_masks):
        """Clip the length of att_masks and att_feats to the maximum length"""
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def pad_unsort_packed_sequence(self, input, inv_ix):
        """Make variable length sequences equal length and return to original order"""
        tmp, _ = pad_packed_sequence(input, batch_first=True)
        tmp = tmp[inv_ix]
        return tmp

    def sort_pack_padded_sequence(self, input, lengths):
        """Sort sequences by length for padding"""
        sorted_lengths, indices = torch_sort(lengths, descending=True)
        tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
        inv_ix = indices.clone()
        inv_ix[indices] = torch_arange(0, len(indices)).type_as(inv_ix)
        return tmp, inv_ix

    def pack_wrapper(self, module, att_feats, att_masks):
        """Apply embedding to padded inputs"""
        if att_masks is not None:
            packed, inv_ix = self.sort_pack_padded_sequence(
                att_feats, att_masks.data.long().sum(1)
            )
            return self.pad_unsort_packed_sequence(
                PackedSequence(module(packed[0]), packed[1]), inv_ix
            )
        else:
            return module(att_feats)
