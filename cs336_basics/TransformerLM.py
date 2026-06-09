import torch
import torch.nn as nn
from jaxtyping import Float, Int
from cs336_basics.TransformerBlock import TransformerBlock
from cs336_basics.RMSNorm import RMSNorm

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 1e-5
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.eps = eps

        # 词嵌入
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        # Transformer 层堆叠
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta
            ) for _ in range(num_layers)
        ])

        # 最终归一化
        self.ln_final = RMSNorm(d_model, eps)

        # 语言模型头
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        # 1. 词嵌入
        x = self.token_embeddings(x)

        # 2. 过所有 Transformer Block
        for layer in self.layers:
            x = layer(x)

        # 3. 最终归一化
        x = self.ln_final(x)

        # 4. 输出 logits
        logits = self.lm_head(x)

        return logits