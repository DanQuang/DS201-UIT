from torch.utils.data import Dataset
import numpy as np
import torchvision
from torchvision import transforms

class ChessXrayDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = torchvision.datasets.ImageFolder(
            root= data_path,
            transform= transforms.ToTensor()
        )

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image, label = self.data[index]
        return {'image': image, 'label': label}
