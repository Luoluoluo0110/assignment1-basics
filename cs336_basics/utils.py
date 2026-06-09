import torch
import torch.nn as nn
from cs336_basics.rotaryPositionalEmbedding import RotaryPositionalEmbedding
import math
def softmax(x, dim):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shift = x -x_max
    exp_x = torch.exp(x_shift)
    sum_x = torch.sum(exp_x, dim=dim, keepdim=True)
    out = exp_x / sum_x
    return out
    
def  scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.size(-1)
    scale = 1.0 / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    attn_scores = torch.matmul(Q, K.transpose(-2, -1))
    
    attn_scores = attn_scores * scale
    
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
    
    attn_weight = softmax(attn_scores, dim=-1)
    
    output = torch.matmul(attn_weight, V)
    return output


def cross_entropy(x, targets):
    # 这个写法会溢出
    # prob = softmax(x, dim=-1)
    # targets_prob = torch.gather(prob, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # loss = torch.mean(-torch.log(targets_prob))
    # 将最后维的最大数
    max_val = torch.max(x, dim=-1, keepdim=True)[0]

    logsumexp = torch.log(torch.sum(torch.exp(x-max_val),dim=-1, keepdim=True)) + max_val
    x_target = torch.gather(x, dim=-1, index=targets.unsqueeze(-1))
    loss_per_sample = logsumexp - x_target

    return loss_per_sample.mean()


def gradient_clipping(parameters, max_l2_norm: float, eps: float = 1e-6):
    """
    全局 L2 梯度裁剪（安全、可复用、训练/测试通用）
    """
    # 先转成 list，保证可以多次遍历，不会被耗尽
    parameters = list(parameters)

    # 全程用 tensor 累加，不转 Python float
    total_norm = torch.tensor(0.0, device=parameters[0].device if parameters else "cpu")
    
    for p in parameters:
        if p.grad is None:
            continue
        grad_norm = p.grad.norm(2)
        total_norm = total_norm + grad_norm ** 2

    total_norm = torch.sqrt(total_norm)
    clip_coef = max_l2_norm / (total_norm + eps)

    # 裁剪梯度（原地操作）
    if clip_coef < 1:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.mul_(clip_coef)
def lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it < cosine_cycle_iters:
        # progress = (it - Tw) / (Tc - Tw)
        # cosine = cos(π * progress)
        # lr = min_lr + 0.5*(max-min) * (1 + cosine)
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine = math.cos(math.pi * progress)
        lr = min_learning_rate + 0.5*(max_learning_rate-min_learning_rate) * (1 + cosine)
        return lr
    
    return min_learning_rate