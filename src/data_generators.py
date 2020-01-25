import torch
import torch.distributions as dist

alpha_dist = dist.Uniform(torch.tensor([0.,0.]), torch.tensor([1.,1.]))
