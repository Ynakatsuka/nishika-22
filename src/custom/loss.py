import kvt
import kvt.losses
import torch.nn as nn


@kvt.LOSSES.register
class SampleLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        raise NotImplementedError
