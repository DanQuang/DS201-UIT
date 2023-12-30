from data_utils import utils
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import torchvision
from torchvision import transforms

class MyDataset(Dataset):
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

class Load_Data_ChestXray:
    def __init__(self, config):
        self.train_batch = config["train_batch"]
        self.dev_batch = config["dev_batch"]
        self.test_batch = config["test_batch"]

        self.train_path = config['dataset']['ChestXray']["train_path"]
        self.dev_path = config['dataset']['ChestXray']["dev_path"]
        self.test_path = config['dataset']['ChestXray']["test_path"]

    def load_train_dev(self):
        train_dataset = MyDataset(self.train_path)
        val_dataset = MyDataset(self.dev_path)

        train_dataloader = DataLoader(train_dataset, self.train_batch, shuffle= True)
        dev_dataloader = DataLoader(val_dataset, self.dev_batch, shuffle= False)

        return train_dataloader, dev_dataloader
    
    def load_test(self):
        test_dataset = MyDataset(self.test_path)
        test_dataloader = DataLoader(test_dataset, self.test_batch, shuffle= False)
        return test_dataloader
    
class Load_Data_Jewellery:
    def __init__(self, config):
        self.train_batch = config["train_batch"]
        self.dev_batch = config["dev_batch"]
        self.test_batch = config["test_batch"]

        self.train_path = config['dataset']['Jewellery']["train_path"]
        self.test_path = config['dataset']['Jewellery']["test_path"]

    def load_train_dev(self):
        train_dataset = MyDataset(self.train_path)
        train_size = int(0.9*len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, self.train_batch, shuffle= True)
        dev_dataloader = DataLoader(val_dataset, self.dev_batch, shuffle= False)

        return train_dataloader, dev_dataloader
    
    def load_test(self):
        test_dataset = MyDataset(self.test_path)
        test_dataloader = DataLoader(test_dataset, self.test_batch, shuffle= False)
        return test_dataloader
    
class Load_Data_Vegetable:
    def __init__(self, config):
        self.train_batch = config["train_batch"]
        self.dev_batch = config["dev_batch"]
        self.test_batch = config["test_batch"]

        self.train_path = config['dataset']['Vegetable']["train_path"]
        self.dev_path = config['dataset']['Vegetable']["dev_path"]
        self.test_path = config['dataset']['Vegetable']["test_path"]

    def load_train_dev(self):
        train_dataset = MyDataset(self.train_path)
        val_dataset = MyDataset(self.dev_path)

        train_dataloader = DataLoader(train_dataset, self.train_batch, shuffle= True)
        dev_dataloader = DataLoader(val_dataset, self.dev_batch, shuffle= False)

        return train_dataloader, dev_dataloader
    
    def load_test(self):
        test_dataset = MyDataset(self.test_path)
        test_dataloader = DataLoader(test_dataset, self.test_batch, shuffle= False)
        return test_dataloader
    
class Load_Data:
    def __init__(self, config):
        self.name_dataset = config["dataset"]["name_dataset"]
        self.load_data = self.build_load_data(config)

    def build_load_data(self, config):
        if self.name_dataset == "ChestXray":
            return Load_Data_ChestXray(config)
        elif self.name_dataset == "Jewellery":
            return Load_Data_Jewellery(config)
        elif self.name_dataset == "Vegetable":
            return Load_Data_Vegetable(config)
        
    def load_train_dev(self):
        return self.load_data.load_train_dev()

    def load_test(self):
        return self.load_data.load_test()