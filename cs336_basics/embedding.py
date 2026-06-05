import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(vocab_size, d_model))

    def forward(self, x):
        return self.weights[x]
    


        
        