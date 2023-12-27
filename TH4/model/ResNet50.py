import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]
        self.pretrained = config["pretrained"]
        self.freeze = config["freeze"]
        self.resnet = models.resnet50(pretrained= self.pretrained)
        if self.freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.resnet.fc = nn.LazyLinear(self.num_classes)

    def forward(self, x):
        if (x.dim() == 3):
            x = x.unsqueeze(dim = 1)
        return self.resnet(x)