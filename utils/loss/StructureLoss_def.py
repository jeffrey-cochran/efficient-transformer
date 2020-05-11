from torch.nn import Module, functional as F
from torch import cat as torch_cat, from_numpy, sum as torch_sum
from utils.loss.rewards import get_scores, get_self_cider_scores


class StructureLoss(Module):
    """
    This loss is inspired by Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018).
    """

    def __init__(self, structure_loss_type=None):
        super(StructureLoss, self).__init__()
        self.loss_type = structure_loss_type

    def forward(self, input, seq, data_gts):
        """
        Input is either logits or log softmax
        """
        out = {}

        batch_size = input.size(0)  # batch_size = sample_size * seq_per_img
        seq_per_img = batch_size // len(data_gts)

        assert seq_per_img == self.opt.train_sample_n, seq_per_img

        mask = (seq > 0).float()
        mask = torch_cat([mask.new_full((mask.size(0), 1), 1), mask[:, :-1]], 1)

        scores = get_scores(data_gts, seq, self.opt)
        scores = from_numpy(scores).type_as(input).view(-1, seq_per_img)
        out["reward"] = scores  # .mean()
        if self.opt.entropy_reward_weight > 0:
            entropy = (
                -(F.softmax(input, dim=2) * F.log_softmax(input, dim=2)).sum(2).data
            )
            entropy = (entropy * mask).sum(1) / mask.sum(1)
            print("entropy", entropy.mean().item())
            scores = scores + self.opt.entropy_reward_weight * entropy.view(
                -1, seq_per_img
            )
        # rescale cost to [0,1]
        costs = -scores
        if self.loss_type == "risk" or self.loss_type == "softmax_margin":
            costs = costs - costs.min(1, keepdim=True)[0]
            costs = costs / costs.max(1, keepdim=True)[0]
        # in principle
        # Only risk need such rescale
        # margin should be alright; Let's try.

        # Gather input: BxTxD -> BxT
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        if self.loss_type == "seqnll":
            # input is logsoftmax
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            target = costs.min(1)[1]
            output = F.cross_entropy(input, target)
        elif self.loss_type == "risk":
            # input is logsoftmax
            input = input * mask
            input = input.sum(1)
            input = input.view(-1, seq_per_img)

            output = (F.softmax(input.exp()) * costs).sum(1).mean()

            # test
            # avg_scores = input
            # probs = F.softmax(avg_scores.exp_())
            # loss = (probs * costs.type_as(probs)).sum() / input.size(0)
            # print(output.item(), loss.item())

        elif self.loss_type == "max_margin":
            # input is logits
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input).max(1)[0] / 2
            output = output.mean()

            # sanity test
            # avg_scores = input + costs
            # scores_with_high_target = avg_scores.clone()
            # scores_with_high_target.scatter_(1, costs.min(1)[1].view(-1, 1), 1e10)

            # target_and_offender_index = scores_with_high_target.sort(1, True)[1][:, 0:2]
            # avg_scores = avg_scores.gather(1, target_and_offender_index)
            # target_index = avg_scores.new_zeros(avg_scores.size(0), dtype=torch.long)
            # loss = F.multi_margin_loss(avg_scores, target_index, size_average=True, margin=0)
            # print(loss.item() * 2, output.item())

        elif self.loss_type == "multi_margin":
            # input is logits
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input)
            output = output.mean()

            # sanity test
            # avg_scores = input + costs
            # loss = F.multi_margin_loss(avg_scores, costs.min(1)[1], margin=0)
            # print(output, loss)

        elif self.loss_type == "softmax_margin":
            # input is logsoftmax
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target)

        elif self.loss_type == "real_softmax_margin":
            # input is logits
            # This is what originally defined in Kevin's paper
            # The result should be equivalent to softmax_margin
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target)

        elif self.loss_type == "new_self_critical":
            """
            A different self critical
            Self critical uses greedy decoding score as baseline;
            This setting uses the average score of the rest samples as baseline
            (suppose c1...cn n samples, reward1 = score1 - 1/(n-1)(score2+..+scoren) )
            """
            baseline = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1] - 1)
            scores = scores - baseline
            # self cider used as reward to promote diversity (not working that much in this way)
            if getattr(self.opt, "self_cider_reward_weight", 0) > 0:
                _scores = get_self_cider_scores(data_gts, seq, self.opt)
                _scores = from_numpy(_scores).type_as(scores).view(-1, 1)
                _scores = _scores.expand_as(scores - 1)
                scores += self.opt.self_cider_reward_weight * _scores
            output = -input * mask * scores.view(-1, 1)
            output = torch_sum(output) / torch_sum(mask)

        out["loss"] = output
        return out
