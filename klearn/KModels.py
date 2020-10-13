import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class ThetaPL(nn.Module):
    def __init__(self, support_size, initial_k=None):
        super(ThetaPL, self).__init__()

        if initial_k:
            self.uniform = initial_k
        else:
            self.uniform = [1. / support_size] * support_size

        self.k = torch.tensor(self.uniform, requires_grad=True).float()
        self.alpha = torch.ones(1, 1, requires_grad=True).float()
        self.beta = torch.ones(1, 1, requires_grad=True).float()

    def forward(self, t, s1, s2):
        red1 = (s1 * s1.log()).sum(dim=1)
        rel1 = -(s1 * (s1 / t).log()).sum(dim=1)

        epsilon = 0.00001
        epsilon_uniform = epsilon * torch.tensor(self.uniform).float()

        K_x = F.softmax(self.k, dim=0)
        K = K_x + epsilon_uniform  # Numerical stability
        inf1 = (s1 * (s1 / K).log()).sum(dim=1)
        theta_x1 = red1 + rel1 + inf1
        red2 = (s2 * s2.log()).sum(dim=1)
        rel2 = -(s2 * (s2 / t).log()).sum(dim=1)
        inf2 = (s2 * (s2 / K).log()).sum(dim=1)
        theta_x2 = red2 + rel2 + inf2
        return torch.sigmoid(10 * (theta_x1 - theta_x2))
