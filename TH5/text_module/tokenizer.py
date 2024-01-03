import torch
from torch import nn 
from data_utils.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence

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

        padding_sequence = pad_sequence(sequence_vectors, True, self.vocab.pad_token_idx())[:, :self.max_length, :]
        padding_sequence = self.fc(padding_sequence)
        return padding_sequence