import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in))
        
    def forward(self, x):
        return x @ self.weight.T
