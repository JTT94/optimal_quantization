import torch


def l2_cost2(x, y, axis=1):
    return torch.sum((x-y)**2, axis=axis).squeeze()


def l2_cost(x, y, axis=1):
    return torch.sqrt(l2_cost2(x, y, axis=axis))


def batch_l2_cost2(xs, ys, collapsed = True):
    batch_size = len(xs)
    num_atoms = len(ys)

    ys_reshape = ys.repeat((batch_size, 1, 1))
    xs_reshape = xs.repeat((num_atoms, 1, 1)).permute((1, 0, 2))

    if collapsed:
        return torch.sum((xs_reshape - ys_reshape) ** 2, axis=[0, 2]) / batch_size
    else:
        return (xs_reshape - ys_reshape) ** 2 / batch_size
