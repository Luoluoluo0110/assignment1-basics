import torch
import torch.nn as nn
from cs336_basics.linear import Linear

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        self.silu = SiLU()
    def forward(self, x):
        gate = self.silu(self.w1(x))
        up = self.w3(x)
        hidden = gate * up
        out = self.w2(hidden)
        return out
    
    