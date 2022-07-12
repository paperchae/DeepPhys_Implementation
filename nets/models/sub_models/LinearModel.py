import torch
import torch.nn as nn
from torch.nn import Module

class LinearModel(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(5184, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, inputs):
        l1 = self.linear1(inputs)
        l2 = self.linear2(l1)

        return l2

