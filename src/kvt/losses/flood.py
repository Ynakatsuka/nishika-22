import torch
import torch.nn as nn


class FloodingBCEWithLogitsLoss(nn.Module):
    def __init__(self, b=0.01, **kwargs):
        super().__init__()
        self.b = b
        self.loss = torch.nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, input, target):
        return torch.abs(self.loss(input, target) - self.b) + self.b
