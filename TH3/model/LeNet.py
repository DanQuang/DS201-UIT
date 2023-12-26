import torch
import torch.nn as nn
from torch.nn import functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features= 400,
                             out_features= 120)
        self.fc2 = nn.Linear(in_features= 120,
                             out_features= 84)
        self.fc3 = nn.Linear(in_features= 84,
                             out_features= num_classes)
        

    def forward(self, x: torch.Tensor):
        x=x.unsqueeze(dim=1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.avgpool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.avgpool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x
