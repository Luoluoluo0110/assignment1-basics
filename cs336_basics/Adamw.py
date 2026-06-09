import torch
import torch.nn as nn
import math
class Adamw(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        # 1. 做检查 lr >=0
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        # 2. 构造 defaults 字典
        defaults = {"lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay
        }
        # 3. 调用父类 __init__
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        # 遍历每个 group
        for group in self.param_groups:
            # 拿出 lr, beta1, beta2, eps, weight_decay
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            # 遍历每个参数 p
            for p in group["params"]:
                if p.grad is None: continue
                
                grad = p.grad
                # 每个 p 的 state = 这个参数自己的 “小账本”
                # 里面只存 3 样东西：
                # m = 这个参数过去梯度的记忆（一阶动量）
                # v = 这个参数过去梯度平方的记忆（二阶动量）
                # t = 这个参数更新到第几步了
                state = self.state[p]
                
                # 如果 state 为空，初始化 t, m, v
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["t"] = 0

                # 取出 m, v, t
                m = state["m"]
                v = state["v"]
                t = state["t"]
                
                # t + 1
                t = state["t"]
                t += 1
                state["t"] = t
                # 更新 m
                # mₜ = β₁ × mₜ₋₁ + (1 − β₁) × gₜ
                m.mul_(beta1).add_(grad, alpha=1-beta1)              
                # 更新 v
                # v = β₂ * v + (1 − β₂) * (梯度)²
                v.mul_(beta2).add_(grad.pow(2), alpha=1-beta2)
                # 计算偏差修正 m_hat, v_hat
                m_hat = m / (1 - beta1 ** t) 
                v_hat = v / (1 - beta2 ** t)
                # 计算 update, 平均抖动除以抖动大小（衡量方向的参数）
                update = m_hat / (torch.sqrt(v_hat) + eps)

                # ====================
                # AdamW 关键：
                # 1. 权重衰减 p 更新
                # 2. 主更新 p 更新
                # ====================
                # p = p - (lr * wd * p) - (lr * update) 多出的第一步是为了让权重变小
                p.data -= lr * weight_decay * p.data
                p.data -= lr * update