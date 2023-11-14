from torch.utils.data import Dataset
import numpy as np
import idx2numpy

class FashionMNISTDataset(Dataset):
    def __init__(self, img_path, label_path):
        super().__init__()

        imgs = idx2numpy.convert_from_file(img_path)
        labels = idx2numpy.convert_from_file(label_path)

        self.__data = {}

        for idx, (img, label) in enumerate(zip(imgs, labels)):
            self.__data[idx] = {
                "image": img,
                "label": label
            }

    def __len__(self):
        return len(self.__data)
    
    def __getitem__(self, idx):
        return self.__data[idx]