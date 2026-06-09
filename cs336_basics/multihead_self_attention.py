import torch
import torch.nn as nn
from cs336_basics import utils
from jaxtyping import Float

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def split_head(self, x):
        B, S, _ = x.shape
        x = x.view(B, S, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # [B, H, S, d_k]
 
    def concat_head(self, x):
        B, H, S, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, S, self.d_model)

    def forward(self, in_features):
        Q = in_features @ self.q_proj.weight.T
        K = in_features @ self.k_proj.weight.T
        V = in_features @ self.v_proj.weight.T

        Q = self.split_head(Q)
        K = self.split_head(K)
        V = self.split_head(V)

        B, _, S, _ = Q.shape
        mask = torch.tril(torch.ones(S, S, device=in_features.device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        attn_output = utils.scaled_dot_product_attention(Q, K, V, mask)
        output = self.concat_head(attn_output)
        output = output @ self.o_proj.weight.T
        return output
