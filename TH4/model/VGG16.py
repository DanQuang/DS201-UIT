import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class VGG16(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]
        self.pretrained = config["pretrained"]
        self.freeze = config["freeze"]
        self.vgg16 = models.vgg16(weights= 'DEFAULT')
        if self.freeze:
            for param in self.vgg16.parameters():
                param.requires_grad = False

        self.vgg16.classifier[-1] = nn.LazyLinear(self.num_classes)

    def forward(self, x):
        if (x.dim() == 3):
            x = x.unsqueeze(dim = 1)
        return self.vgg19(x)