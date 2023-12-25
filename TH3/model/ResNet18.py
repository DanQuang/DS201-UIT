import torch
from torch import nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, num_channels, use_1x1conv= False, strides = 1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size= 3, padding= 1, stride= strides)
        self.bn1 = nn.LazyBatchNorm2d()
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size= 3, padding= 1)
        self.bn2 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size= 1, stride= strides)
        else:
            self.conv3 = None
    
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    

class ResNet(nn.Module):
    def __init__(self, arch, num_classes= 10):
        super().__init__()
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f"b{i + 2}", self.block(*b, first_block= (i == 0)))
        self.net.add_module("last", nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)
        ))

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size= 7, stride= 2, padding= 3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
        )
    
    def block(self, num_residuals, num_channels, first_block= False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels= num_channels, use_1x1conv= True, strides= 2))
            else:
                blk.append(Residual(num_channels= num_channels))
        return nn.Sequential(*blk)
    
class ResNet18(ResNet):
    def __init__(self, num_classes = 10):
        super.__init__(((2, 64), (2, 128), (2, 256), (2, 512)), num_classes)

    def forward(self, x):
        if (x.dim() == 3):
            x = x.unsqueeze(dim = 1)
        return self.net(x)