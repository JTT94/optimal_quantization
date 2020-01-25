import torch


def soft_exponential_cost(x, ys, g, cost_func, epsilon):
    cost_vector = cost_func(x, ys)
    return torch.exp((g - cost_vector)/epsilon)


def normalised_soft_exponential_cost(x, ys, g, cost_func, epsilon):
    cost_vector = cost_func(x, ys)
    cost_vector = torch.exp((g - cost_vector) / epsilon)
    total = torch.sum(cost_vector)

    return cost_vector / total


def soft_c_transform(x, ys, b, g, cost_func, epsilon):

    exponential_cost_vector = soft_exponential_cost(x, ys, g, cost_func, epsilon)
    return -epsilon * torch.log(torch.dot(exponential_cost_vector, b))


def stochastic_dual_approx(x, ys, b, g, cost_func, epsilon):
    approx = soft_c_transform(x, ys, b, g, cost_func, epsilon) + torch.dot(g, b)
    return approx
