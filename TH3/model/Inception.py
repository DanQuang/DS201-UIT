import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()

        self.p1 = nn.Conv2d(in_channels= in_channels, kernel_size= 1)
        
        self.p2_1 = nn.Conv2d(in_channels= in_channels, kernel_size= 1)
        self.p2_2 = nn.Conv2d(in_channels= in_channels, kernel_size= 3, padding= 1)

        self.p3_1 = nn.Conv2d(in_channels= in_channels, kernel_size= 1)
        self.p3_2 = nn.Conv2d(in_channels= in_channels, kernel_size= 5, padding= 2)

        self.p4_1 = nn.MaxPool2d(kernel_size= 3, padding= 1)
        self.p4_2 = nn.Conv2d(in_channels= in_channels, kernel_size= 1)

    def forward(self, x: torch.Tensor):
        p1 = self.p1(x)
        
        p2 = self.p2_1(x)
        p2 = F.relu(p2)
        p2 = self.p2_2(p2)
        p2 = F.relu(p2)

        p3 = self.p3_1(x)
        p3 = F.relu(p3)
        p3 = self.p3_2(p3)
        p3 = F.relu(p3)

        p4 = self.p4_1(x)
        p4 = self.p4_2(p4)
        p4 = self.relu(p4)

        return torch.concat([p1, p2, p3, p4], dim= 1)