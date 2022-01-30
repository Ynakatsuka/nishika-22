import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.normalize(x)
        return x
