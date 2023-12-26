from torch.utils.data import Dataset
import numpy as np
import torchvision
from torchvision import transforms

class ChessXrayDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data = torchvision.datasets.ImageFolder(
            root= data_path,
            transform= self.transform
        )

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image, label = self.data[index]
        return {'image': image, 'label': label}
