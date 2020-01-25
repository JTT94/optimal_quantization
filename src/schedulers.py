from torch.optim.lr_scheduler import StepLR

def scheduler_constructor(optimizer, gamma):
    return StepLR(optimizer, step_size=1, gamma=gamma)

