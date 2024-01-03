import os
import numpy as np
import pandas as pd

class Vocab:
    def __init__(self, config):
        self.word_to_idx = {}
        self.idx_to_word = {}

        self.dataset_folder = config["dataset"]["dataset_folder"]
        self.train_path = config["dataset"]["train_path"]
        self.valid_path = config["dataset"]["valid_path"]
        self.test_path = config["dataset"]["test_path"]

        self.build_vocab()

    def all_word(self):
        train = pd.read_csv(os.path.join(self.dataset_folder, self.train_path), encoding= 'utf-8')
        valid = pd.read_csv(os.path.join(self.dataset_folder, self.valid_path), encoding= 'utf-8')
        test = pd.read_csv(os.path.join(self.dataset_folder, self.test_path), encoding= 'utf-8')

        word_counts = {}

        for data_file in [train, valid, test]:
            for item in data_file["sentence"]:
                for word in item.split():
                    if word not in word_counts:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1

        self.special_token = ['<UNK>', '<CLS>', '<SEP>']
        for w in self.special_token:
            if w not in word_counts:
                word_counts[w] = 1
            else:
                word_counts[w] += 1

        sorted_word_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))
        all_word = list(sorted_word_counts.keys())         

        return all_word, sorted_word_counts       

    def build_vocab(self):
        all_word, _ = self.all_word()
        self.word_to_idx = {word: idx + 1 for word, idx in enumerate(all_word)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
    
    def convert_tokens_to_ids(self, tokens):
        return [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        return [self.idx_to_word[idx] for idx in ids]
    
    def vocab_size(self):
        return len(self.word_to_idx) + 1
    
    def pad_token_idx(self):
        return 0