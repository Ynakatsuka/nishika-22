import torch
import torch.nn as nn


class MultiSampleDropout(nn.Module):
    def __init__(self, in_features, num_classes, p=0.5, n_multi_samples=5):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.dropouts = nn.ModuleList(
            [nn.Dropout(p) for _ in range(n_multi_samples)]
        )

    def forward(self, x):
        x = torch.mean(
            torch.stack(
                [self.fc(dropout(x)) for dropout in self.dropouts], dim=0
            ),
            dim=0,
        )
        return x
