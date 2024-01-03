import torch
from torch import nn 
from data_utils.vocab import Vocab

class CountVectorizer(nn.Module):
    def __init__(self, config):
        super(CountVectorizer, self).__init__()
        self.vocab = Vocab(config)
        self.fc = nn.LazyLinear(config["embedding"]["d_model"])

    def forward(self, input_texts):
        count_vectors = []
        for input_text in input_texts:
            word_counts = torch.zeros(self.vocab.vocab_size())
            for word in input_text.split():
                word_counts[self.vocab.word_to_idx.get(word, self.vocab.word_to_idx['<UNK>'])] += 1
            count_vectors.append(word_counts)

        count_vectors = torch.stack(count_vectors, dim = 0) # Hợp các count vector thành tensor
        count_vectors = self.fc(count_vectors)
        return count_vectors