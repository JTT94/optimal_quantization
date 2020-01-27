from torch.optim.lr_scheduler import StepLR


def scheduler_constructor(gamma):
    return lambda optimizer: StepLR(optimizer, step_size=1, gamma=gamma)

