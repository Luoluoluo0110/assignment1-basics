import torch
import torch.nn as nn
import cs336_basics.utils as utils
from cs336_basics.swiglu import SwiGLU
from cs336_basics.multihead_self_attention_with_rope import MultiheadSelfAttentionWithRoPE
from cs336_basics.RMSNorm import RMSNorm


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.eps = 1e-5

        # 注意力层参数
        self.attn = MultiheadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
        # 归一化
        self.ln1 = RMSNorm(d_model, eps=1e-5)
        self.ln2 = RMSNorm(d_model, eps=1e-5)

        # FFN (SwiGLU)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, in_features):
        x = in_features
        B, S, _ = x.shape
        token_positions = torch.arange(S, device=x.device).unsqueeze(0).repeat(B, 1)

        # 注意力模块
        attn_norm = self.ln1(x)
        attn_out = self.attn(attn_norm, token_positions)
        x = x + attn_out

        # FFN 模块
        ffn_norm = self.ln2(x)
        ffn_out = self.ffn(ffn_norm)
        x = x + ffn_out

        return x
