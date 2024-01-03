from torch.utils.data import Dataset, DataLoader

from data_utils.vocab import Vocab
from data_utils.utils import preprocess_sentence
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, data_path, vocab= None):
        super(MyDataset, self).__init__()
        data = pd.read_csv(data_path, encoding= 'utf-8')
        self.sentences = []
        self.sentiments = []

        self.vocab = vocab

        for sample in data:
            self.sentences.append(sample["sentence"])
            self.sentiments.append(sample["sentiment"])
        

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.sentences[idx], self.sentiments[idx]
    
class Load_data:
    def __init__(self, config):
        self.train_batch = config["train_batch"]
        self.dev_batch = config["valid_batch"]
        self.test_batch = config["test_batch"]

        self.train_path = config['dataset']["train_path"]
        self.dev_path = config['dataset']["valid_path"]
        self.test_path = config['dataset']["test_path"]

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