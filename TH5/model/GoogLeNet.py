import torch
import torch.nn as nn
from torch.nn import functional as F
from model.Inception import Inception

class GooLeNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 64, kernel_size= 7, padding= 3, stride= 2)
        self.conv2 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 1)
        self.conv3 = nn.Conv2d(in_channels= 64, out_channels= 192, kernel_size= 3, padding= 1)
        self.maxpool = nn.MaxPool2d(3, stride= 2, padding= 1)

        self.inception_1 = Inception(in_channels= 192)
        self.inception_2 = Inception()


    def forward(self, x: torch.Tensor):
        pass