import torch
from src.utils import uniform_weights
from src.discrete_entropic_reg import stochastic_dual_approx
from src.ground_cost import batch_l2_cost2

class Quantizer(object):

    def __init__(self, init_g, init_ys, epsilon, ground_cost=batch_l2_cost2, init_b=None):
        self.num_atoms = len(init_ys)

        # trainable atoms and map
        self.g = torch.tensor(init_g, requires_grad=True)
        self.ys = torch.tensor(init_ys, requires_grad=True)
        self.epsilon = torch.tensor(epsilon)

        if init_b is None:
            self.b = uniform_weights(self.num_atoms)
        else:
            self.b = torch.tensor(init_g, requires_grad=False)

        self.ground_cost = ground_cost

    def stochastic_dual_approx(self, xs):
        return stochastic_dual_approx(xs, self.ys, self.b, self.g, cost_func=self.ground_cost, epsilon=self.epsilon)
