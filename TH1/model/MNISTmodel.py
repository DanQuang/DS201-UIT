import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.MLP = nn.Linear(in_features= 784,
                                 out_features= 10)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.MLP(x)
        return x