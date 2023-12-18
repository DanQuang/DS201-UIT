import torch
import torch.nn as nn
import torch.nn.init as init

class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.MLP1 = nn.Linear(in_features= 784,
                            out_features= 512)
        self.MLP2 = nn.Linear(in_features= 512,
                              out_features= 256)
        self.MLP3 = nn.Linear(in_features= 256,
                              out_features= 10)
        
        # # W and b use Ones init
        # self.init_weights_1()

        # # W use Xavier and b use Ones
        # self.init_weights_2()

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.MLP1(x)
        x = self.relu(x)
        x = self.MLP2(x)
        x = self.relu(x)
        x = self.MLP3(x)
        return x
    
    def l2_regularization(self, lambda_l2= 0.01):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)

        return lambda_l2 * l2_loss
    
    def init_weights_1(self):
        init.ones_(self.MLP1.weight)
        init.ones_(self.MLP1.bias)
        init.ones_(self.MLP2.weight)
        init.ones_(self.MLP2.bias)
        init.ones_(self.MLP3.weight)
        init.ones_(self.MLP3.bias)

    def init_weights_2(self):
        init.xavier_uniform_(self.MLP1.weight)
        init.zeros_(self.MLP1.bias)
        init.xavier_uniform_(self.MLP2.weight)
        init.zeros_(self.MLP2.bias)
        init.xavier_uniform_(self.MLP3.weight)
        init.zeros_(self.MLP3.bias)