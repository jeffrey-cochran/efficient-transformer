from torch.optim import RMSprop, Adagrad, SGD, Adam


def build_optimizer(
    params,
    core_optimizer="sgd",
    learning_rate=None,
    optimizer_alpha=None,
    optimizer_beta=None,
    optimizer_epsilon=None,
    weight_decay=None,
):
    if core_optimizer == "rmsprop":
        return RMSprop(
            params,
            learning_rate,
            optimizer_alpha,
            optimizer_epsilon,
            weight_decay=weight_decay,
        )
    elif core_optimizer == "adagrad":
        return Adagrad(params, learning_rate, weight_decay=weight_decay)
    elif core_optimizer == "sgd":
        return SGD(params, learning_rate, weight_decay=weight_decay)
    elif core_optimizer == "sgdm":
        return SGD(params, learning_rate, optimizer_alpha, weight_decay=weight_decay)
    elif core_optimizer == "sgdmom":
        return SGD(
            params,
            learning_rate,
            optimizer_alpha,
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif core_optimizer == "adam":
        return Adam(
            params,
            learning_rate,
            (optimizer_alpha, optimizer_beta),
            optimizer_epsilon,
            weight_decay=weight_decay,
        )
    else:
        raise Exception("Incorrect optimizerization option")
