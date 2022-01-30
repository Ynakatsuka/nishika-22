import torch.nn as nn


class MultiHead(nn.Module):
    def __init__(
        self, in_features, num_classes_list, head_names, dropout_rate=0.0
    ):
        super().__init__()
        assert len(head_names) == len(num_classes_list)
        self.fc = [
            nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(in_features, num_classes),
            )
            for num_classes in num_classes_list
        ]

    def forward(self, x):
        output = {}
        for key, fc in zip(self.head_names, self.fc):
            output[key] = fc(x)
        return output
