from model.LabelSmoothing_def import LabelSmoothing
from torch.autograd import Variable
from torch import FloatTensor, LongTensor

#
# Set label smoothing parameters
class Loss:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        print(loss.data)
        return loss.data.item() * norm
