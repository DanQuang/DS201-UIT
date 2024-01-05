import torch
from torch import nn 
import torch.nn.functional as F
from text_module.init_embedding import build_text_embbeding

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()

        self.num_labels = config["num_labels"]
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout = config["model"]["dropout"]
        self.text_embedding = build_text_embbeding(config)
        self.rnn = nn.RNN(self.intermediate_dims, self.intermediate_dims,
                          num_layers=config['model']['num_layers'],dropout=self.dropout)
        
        self.fc = nn.LazyLinear(self.num_labels)

    def forward(self, texts):
        embbed = self.text_embedding(texts)
        rnn_output, _ = self.rnn(embbed)
        logits = self.fc(rnn_output[:, -1, :])
        return logits