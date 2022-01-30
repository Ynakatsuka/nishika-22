import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt(self.loss_fn(pred, target))
