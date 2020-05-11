from torch.nn import Module
from torch import sum as torch_sum


class LanguageModelCriterion(Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        #
        # Why would dim be 3? and why reduce to 2?
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        #
        # Truncate to the same size
        target = target[:, : input.size(1)]
        mask = mask[:, : input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        #
        # # Average over each token
        output = torch_sum(output) / torch_sum(mask)

        return output
