import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class VGG19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.vgg19 = models.vgg19(pretrained= True)
        for param in self.vgg19.parameters():
            param.requires_grad = True

        self.vgg19.classifier[-1] = nn.LazyLinear(num_classes)

    def forward(self, x):
        if (x.dim() == 3):
            x = x.unsqueeze(dim = 1)
        return self.vgg19(x)