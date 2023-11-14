import torch
import torch.nn as nn

class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.MLP1 = nn.Linear(in_features= 784,
                            out_features= 512)
        self.MLP2 = nn.Linear(in_features= 512,
                              out_features= 256)
        self.MLP3 = nn.Linear(in_features= 256,
                              out_features= 10)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.MLP1(x)
        x = self.relu(x)
        x = self.MLP2(x)
        x = self.relu(x)
        x = self.MLP3(x)
        return x