from torch.nn import Module
from torch import cat as torch_cat, sum as torch_sum


class RewardCriterion(Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        print(f"INPUT")

        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = (seq > 0).float()
        mask = torch_cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(
            -1
        )
        output = -input * reward * mask
        output = torch_sum(output) / torch_sum(mask)

        return output
