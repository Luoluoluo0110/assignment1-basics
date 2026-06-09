import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    # RMS -> RMS(x) = sqrt( mean(x²) + eps )
    # RMSNorm(x) = x / RMS(x) * g

    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight             
        
        