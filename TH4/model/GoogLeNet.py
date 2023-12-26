import torch
from torch import nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # branch 1
        self.b1 = nn.Sequential(
            nn.LazyConv2d(c1, kernel_size= 1),
            nn.BatchNorm2d(c1)
        )

        # branch 2
        self.b2 = nn.Sequential(
            nn.LazyConv2d(c2[0], kernel_size= 1),
            nn.BatchNorm2d(c2[0]),
            nn.ReLU(),
            nn.LazyConv2d(c2[1], kernel_size= 3, padding= 1),
            nn.BatchNorm2d(c2[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # branch 3
        self.b3 = nn.Sequential(
            nn.LazyConv2d(c3[0], kernel_size= 1),
            nn.BatchNorm2d(c3[0]),
            nn.ReLU(),
            nn.LazyConv2d(c3[1], kernel_size= 5, padding= 2),
            nn.BatchNorm2d(c3[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # branch 4
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size= 3, stride= 1, padding= 1),
            nn.LazyConv2d(c4, kernel_size= 1),
            nn.BatchNorm2d(c4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)
        return torch.cat([b1, b2, b3, b4], dim = 1)
    

class GoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
                                 nn.LazyLinear(self.num_classes))
        
    def forward(self, x):
        if (x.dim() == 3):
            x = x.unsqueeze(dim = 1)
        return self.net(x)

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size= 7, stride= 2, padding= 3),
            nn.ReLU(), nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
        )
    
    def b2(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size= 1), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size= 3, padding= 1), nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
        )
    
    def b3(self):
        return nn.Sequential(
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
        )
    
    def b4(self):
        return nn.Sequential(
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
        )
    
    def b5(self):
        return nn.Sequential(
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten()
        )