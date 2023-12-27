from data_utils import utils
from torch.utils.data import DataLoader, Dataset, random_split
from data_utils import dataset

class Load_Data_ChestXray:
    def __init__(self, config):
        self.train_batch = config["train_batch"]
        self.dev_batch = config["dev_batch"]
        self.test_batch = config["test_batch"]

        self.train_path = config["train_path"]
        self.val_path = config["val_path"]
        self.test_path = config["test_path"]

    def load_train_dev(self):
        train_dataset = dataset.ChessXrayDataset(self.train_path)
        val_dataset = dataset.ChessXrayDataset(self.val_path)


        train_dataloader = DataLoader(train_dataset, self.train_batch, shuffle= True)
        dev_dataloader = DataLoader(val_dataset, self.dev_batch, shuffle= False)

        return train_dataloader, dev_dataloader
    
    def load_test(self):
        test_dataset = dataset.ChessXrayDataset(self.test_path)
        test_dataloader = DataLoader(test_dataset, self.test_batch, shuffle= False)
        return test_dataloader