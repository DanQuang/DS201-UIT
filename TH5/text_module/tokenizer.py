import torch
from torch import nn 
from data_utils.vocab import Vocab
from keras.utils import pad_sequences

class Tokenizer(nn.Module):
    def __init__(self, config):
        super(Tokenizer, self).__init__()
        self.vocab = Vocab(config)
        self.max_length = config['text_embedding']['max_length']
        self.fc = nn.LazyLinear(config["text_embedding"]["d_model"])

    def forward(self, input_texts):
        sequence_vectors = []
        for input_text in input_texts:
            tokens_to_ids = torch.tensor(self.vocab.convert_tokens_to_ids(input_text.split()))
            sequence_vectors.append(tokens_to_ids)

        padding_sequence = pad_sequences(sequence_vectors, value= self.vocab.pad_token_idx(), maxlen= self.max_length, padding= "post")
        padding_sequence = self.fc(torch.tensor(padding_sequence, dtype= torch.float32).to(self.fc.weight.device))
        return padding_sequence