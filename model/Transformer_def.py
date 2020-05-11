#
# Torch imports
from torch.nn import Sequential, Linear, ReLU, Dropout, Module
from torch import long as torch_long, zeros as torch_zeros, max as torch_max
from torch import cat as torch_cat
from torch import sort as torch_sort
from torch import arange as torch_arange
from torch import from_numpy
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

#
# External imports
from numpy import isin

#
# Local imports
from utils.constants import GREEDY, BEAM
from utils.masks import subsequent_mask
from utils.misc import repeat_tensors
from utils.model import make_model


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
        resnet_conv_feature_size=None,
    ):
        #
        # Initialize super model
        super(Transformer, self).__init__()

        #
        # Set hyperparams
        self.num_layers = N
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.dropout = dropout
        self.resnet_conv_feature_size = resnet_conv_feature_size

        #
        # Function for data embedding
        self.embed = lambda x: x
        self.fc_embed = lambda x: x
        self.att_embed = Sequential(
            Linear(self.resnet_conv_feature_size, self.d_model),
            ReLU(),
            Dropout(dropout),
        )

        #
        # Note that vocab size is increased by one for END token
        self.model = make_model(
            0,
            tgt_vocab_size + 1,
            N=self.num_layers,
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
        att_feats, seq, att_masks, seq_mask = self.prepare_feature_for_forward_prop(
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

    def sample(
        self,
        fc_feats,
        att_feats,
        att_masks=None,
        sample_method=GREEDY,
        temperature=1.0,
        sample_n=1,
        output_logsoftmax=1,
        decoding_constraint=0,
        block_trigrams=0,
        remove_bad_endings=0,
    ):
        #
        # Beam search not implemented
        # ===================
        # beam_size=1,
        # group_size=1,
        # if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
        #     return self._sample_beam(fc_feats, att_feats, att_masks, opt)
        # if group_size > 1:
        #     return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)

        (
            p_fc_feats,
            p_att_feats,
            pp_att_feats,
            p_att_masks,
        ) = self.prepare_feature_for_generation(fc_feats, att_feats, att_masks)

        #
        # Don't implemnt diverse sampling
        # ===================
        # if sample_n > 1:
        #     p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = repeat_tensors(
        #         sample_n, [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
        #     )

        # trigrams = []  # will be a list of batch_size dictionaries

        #
        # Sampled words
        seq = fc_feats.new_zeros(
            (batch_size * sample_n, self.seq_length), dtype=torch_long
        )

        #
        # Log prob of words
        seqLogprobs = fc_feats.new_zeros(
            batch_size * sample_n, self.seq_length, self.vocab_size + 1
        )

        for t in range(self.seq_length + 1):

            #
            # Beginning of sentence. No previous words sampled
            if t == 0:
                selected_word_indices = fc_feats.new_zeros(
                    batch_size * sample_n, dtype=torch_long
                )

            #
            # Log prob of vocab
            logprobs, state = self.get_logprobs_state(
                selected_word_indices,
                p_fc_feats,
                p_att_feats,
                pp_att_feats,
                p_att_masks,
                state,
                output_logsoftmax=output_logsoftmax,
            )

            # if decoding_constraint and t > 0:
            #     tmp = logprobs.new_zeros(logprobs.size())
            #     tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float("-inf"))
            #     logprobs = logprobs + tmp

            #
            # Can't sample bad endings (is this right?)
            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                #
                # See if bad ending was just chosen
                prev_bad = isin(seq[:, t - 1].data.cpu().numpy(), self.bad_endings_ix)
                #
                # If last chosen word was bad ending can't choose again?
                # This doesn't seem like it will prevent a bad ending...
                tmp[from_numpy(prev_bad.astype("uint8")), 0] = float("-inf")
                logprobs = logprobs + tmp

            #
            # Trigrams not implemented
            # ============
            # if block_trigrams and t >= 3:
            #     # Store trigram generated at last step
            #     prev_two_batch = seq[:, t - 3 : t - 1]
            #     for i in range(batch_size):  # = seq.size(0)
            #         prev_two = (
            #             prev_two_batch[i][0].item(),
            #             prev_two_batch[i][1].item(),
            #         )
            #         current = seq[i][t - 1]
            #         if t == 3:  # initialize
            #             trigrams.append(
            #                 {prev_two: [current]}
            #             )  # {LongTensor: list containing 1 int}
            #         elif t > 3:
            #             if prev_two in trigrams[i]:  # add to list
            #                 trigrams[i][prev_two].append(current)
            #             else:  # create list
            #                 trigrams[i][prev_two] = [current]
            #     # Block used trigrams at next step
            #     prev_two_batch = seq[:, t - 2 : t]
            #     mask = torch_zeros(
            #         logprobs.size(), requires_grad=False
            #     ).cuda()  # batch_size x vocab_size
            #     for i in range(batch_size):
            #         prev_two = (
            #             prev_two_batch[i][0].item(),
            #             prev_two_batch[i][1].item(),
            #         )
            #         if prev_two in trigrams[i]:
            #             for j in trigrams[i][prev_two]:
            #                 mask[i, j] += 1
            #     # Apply mask to log probs
            #     # logprobs = logprobs - (mask * 1e9)
            #     alpha = 2.0  # = 4
            #     logprobs = logprobs + (
            #         mask * -0.693 * alpha
            #     )  # ln(1/2) * alpha (alpha -> infty works best)

            #
            # Break if max length
            if t == self.seq_length:  # skip if we achieve maximum length
                break

            #
            # Sample next word. Since this is a greedy approach without beams,
            # simply take the most probable word
            sampleLogprobs, selected_word_indices = torch_max(logprobs.data, 1)
            selected_word_indices = selected_word_indices.view(-1).long()

            #
            # Stop when all sequences have finished.
            # Encoded <EOS> as index 0.
            if t == 0:
                unfinished = selected_word_indices > 0
            else:
                #
                # This prevents finished sequences from starting again
                unfinished = unfinished * (selected_word_indices > 0)
            #
            # Don't allow restart of finished sequences
            selected_word_indices = selected_word_indices * unfinished.type_as(
                selected_word_indices
            )

            #
            # Update sentences
            seq[:, t] = selected_word_indices
            seqLogprobs[:, t] = logprobs

            #
            # Break if all are finished.
            # If all false, then sum is 0.
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs
