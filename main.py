from model.LabelSmoothing_def import LabelSmoothing
from utils.model import make_model
from utils.training import NoamOpt, run_epoch, data_gen
from utils.Loss_def import Loss
from torch.optim import Adam

V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(
    model.src_embed[0].d_model,
    1,
    400,
    Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
)

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model, Loss(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model, Loss(model.generator, criterion, None)))
