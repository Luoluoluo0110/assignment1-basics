import torch
import torch.nn as nn
class RotaryPositionalEmbedding(nn.Module):
    # theta: RoPE 的固定超参数一般是 10000用来计算位置频率
    # d_k 每个头的维度，向量的维度
    # max_seq_len: int 最大的句子长度
    # 数学原理，通过极坐标证明
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # 生成所有位置序号
        position = torch.arange(max_seq_len, device=device)
        # 生成维度下标
        dim = torch.arange(0, d_k, 2, device=device).float()
        # 控制频率，越往后的维度旋转的越慢
        freq = 1.0 / (theta ** (dim / d_k))

        # 位置序号和频率外积，得到旋转的角度
        pos_freq = torch.outer(position, freq)
        cos_table = torch.cos(pos_freq)
        sin_table = torch.sin(pos_freq)
        self.register_buffer("cos", cos_table)
        self.register_buffer("sin", sin_table)
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        new_x1 = x1 * cos - x2 * sin
        new_x2 = x1 * sin + x2 * cos
        # 拼回去
        out = torch.stack([new_x1, new_x2], dim=-1).flatten(-2)

        return out