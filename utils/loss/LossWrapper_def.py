from torch import from_numpy, no_grad, tensor as torch_tensor
from torch.nn import Module
from utils.constants import MARGIN_KEYWORD, SOFTMAX
from utils.loss.StructureLoss_def import StructureLoss
from utils.loss.RewardCriterion_def import RewardCriterion
from utils.loss.LabelSmoothing_def import LabelSmoothing
from utils.loss.LanguageModelCriterion_def import LanguageModelCriterion
from utils.loss.rewards import get_self_critical_reward


class LossWrapper(Module):
    def __init__(
        self,
        model,
        label_smoothing=None,
        structure_loss_weight=None,
        train_sample_method=None,
        train_beam_size=None,
        struc_use_logsoftmax=None,
        train_sample_n=None,
        structure_loss_type=None,
    ):
        #
        # Init Module()
        super(LossWrapper, self).__init__()

        #
        # Init loss params
        self.model = model
        self.structure_loss_weight = structure_loss_weight
        self.train_sample_method = train_sample_method
        self.train_beam_size = train_beam_size
        self.struc_use_logsoftmax = struc_use_logsoftmax
        self.train_sample_n = train_sample_n
        #
        # Define losses
        if label_smoothing > 0:
            self.crit = LabelSmoothing(smoothing=label_smoothing)
        else:
            self.crit = LanguageModelCriterion()
        self.rl_crit = RewardCriterion()
        self.struc_crit = StructureLoss(structure_loss_type=structure_loss_type)

    def forward(
        self,
        fc_feats,
        att_feats,
        labels,
        masks,
        att_masks,
        gts,
        gt_indices,
        sc_flag,
        struc_flag,
    ):
        #
        # Might want to make dictionary structure explict here
        out = {}
        if struc_flag:
            if self.structure_loss_weight < 1:
                #
                # Compute language model loss with crit( model.forward() )
                # NOTE: without label smoothing, label[] is just a a one-hot
                # encoding that doubles as a probability distribution
                # The output of self.model() is the predicted probabilities
                lm_loss = self.crit(
                    self.model(fc_feats, att_feats, labels, att_masks),
                    labels[..., 1:],
                    masks[..., 1:],
                )
            else:
                #
                # Report lm_loss regardless of whether it is used
                lm_loss = torch_tensor(0).type_as(fc_feats)
            if self.structure_loss_weight > 0:
                #
                # NOTE: this is not a model.forward() method. In original code,
                # _sample and _forward were both hidden within model.forward().
                # Unclear if pytorch needs that for gradient computation here.
                # I think it may not, since this is just used to compute a second
                # loss. I'm wondering if this isn't just a duplication of effort.
                # In other words, if the code were refactored, you may be able to
                # use the output of self.crit() above for sampling and computing the
                # lm_loss
                gen_result, sample_logprobs = self.model.sample(
                    fc_feats,
                    att_feats,
                    att_masks,
                    sample_method=self.train_sample_method,
                    beam_size=self.train_beam_size,
                    output_logsoftmax=(
                        self.struc_use_logsoftmax
                        or self.structure_loss_type == SOFTMAX
                        or not (MARGIN_KEYWORD in self.structure_loss_type)
                    ),
                    sample_n=self.train_sample_n,
                )
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts)
            else:
                #
                # Report structure loss regardless, but set to zero
                struc_loss = {
                    "loss": torch_tensor(0).type_as(fc_feats),
                    "reward": torch_tensor(0).type_as(fc_feats),
                }
            #
            # Treat loss as linear combination of structure loss and lm_loss
            loss = (
                1 - self.structure_loss_weight
            ) * lm_loss + self.structure_loss_weight * struc_loss["loss"]
            #
            # Why is reward in struc_loss?
            out["lm_loss"] = lm_loss
            out["struc_loss"] = struc_loss["loss"]
            out["reward"] = struc_loss["reward"]
        elif not sc_flag:
            #
            # When not self-critical, it's just calling normal forward().
            # This is, in effect, guaranteeing that LossWrapper does not
            # affect the ability of model.forward() to train as it would
            # normally. NOTE: This is just the language model loss
            loss = self.crit(
                self.model(fc_feats, att_feats, labels, att_masks),
                labels[..., 1:],
                masks[..., 1:],
            )
        else:
            #
            # Generate greedy baseline which is subtracted from
            # sampled sentences to reduce variance in reinforcement
            # learning, per the self-critical sequence training (SCST)
            # algorithm
            self.model.eval()
            with no_grad():
                greedy_res, _ = self.model(
                    fc_feats,
                    att_feats,
                    att_masks,
                    mode="sample",
                    opt={
                        "sample_method": self.sc_sample_method,
                        "beam_size": self.sc_beam_size,
                    },
                )
            #
            # Now generate sample probabilities to use for sampling
            # sentences for SCST
            self.model.train()
            gen_result, sample_logprobs = self.model(
                fc_feats,
                att_feats,
                att_masks,
                opt={
                    "sample_method": self.train_sample_method,
                    "beam_size": self.train_beam_size,
                    "sample_n": self.train_sample_n,
                },
                mode="sample",
            )
            gts = [gts[_] for _ in gt_indices.tolist()]
            #
            # Compute self-critical reward from baseline model
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = from_numpy(reward).float().to(gen_result.device)
            #
            # Compute loss from self-critcial reward
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out["reward"] = reward[:, 0].mean()
        out["loss"] = loss
        return out
