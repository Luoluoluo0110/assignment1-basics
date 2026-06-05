import torch
import torch.nn as nn
from cs336_basics.rotaryPositionalEmbedding import RotaryPositionalEmbedding

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

def multihead_self_attention(d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features):
     d_k = d_model // num_heads
     # 多头注意力就是把特征分组，每个组进行注意力机制的计算，然后计算
     # 将批次和每个批次的长度记住
     batch_size, seq_len, _ = in_features.shape
     
     Q = in_features @ q_proj_weight.T
     K = in_features @ k_proj_weight.T
     V = in_features @ v_proj_weight.T
     
     # 拆分多头：[B, S, d_model] → [B, num_head, S, d_k]
     def split_head(x):
        # 将最后一维拆开成两个维度
        x = x.view(batch_size, seq_len, num_heads, d_k)
        # 交换维度
        return x.permute(0, 2, 1, 3)
    
     Q, K, V = split_head(Q), split_head(K), split_head(V)
     
     # causal mask
     mask = torch.tril(torch.ones(seq_len, seq_len))
     # TO BE (1, 1, seq_len, seq_len)
     mask.unsqueeze(0).unsqueeze(0)
     
     # 注意力计算
     output = scaled_dot_product_attention(Q, K, V, mask)
     
     def concat_head(x):
          x = x.permute(0, 2, 1, 3).contiguous()
          return x.view(batch_size, seq_len, d_model)
     
     output = concat_head(output)
     output = output @ o_proj_weight.T
     return output
def multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features, token_positions):
     # 复用代码
     d_k = d_model // num_heads
     batch_size, seq_len, _ = in_features.shape
     Q = in_features @ q_proj_weight.T
     K = in_features @ k_proj_weight.T
     V = in_features @ v_proj_weight.T
     
     # 拆分多头：[B, S, d_model] → [B, num_head, S, d_k]
     def split_head(x):
        # 将最后一维拆开成两个维度
        x = x.view(batch_size, seq_len, num_heads, d_k)
        # 交换维度
        return x.permute(0, 2, 1, 3)
    
     Q, K, V = split_head(Q), split_head(K), split_head(V)
     rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=Q.device)
     Q = rope(Q, token_positions)
     K = rope(K, token_positions)
     mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device))
     mask = mask.unsqueeze(0).unsqueeze(0)    
     output = scaled_dot_product_attention(Q, K, V, mask)
    
     def concat_head(x):
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, d_model)
    
     output = concat_head(output)
     output = output @ o_proj_weight.T
     return output

# −log(softmax(x) target​)≡logsumexp(x)−x target
# 如果你直接写 log(sum(exp(x))) → 溢出
# 所以必须写成 m + log(sum(exp(x-m))) → 稳定
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
