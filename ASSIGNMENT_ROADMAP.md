# CS336 Assignment 1 实现路线图

> 这是一份**规划/检查表**,帮你理清"先写什么、每个要写哪个类和哪些方法、怎么验证"。
> 不含任何解法代码——具体逻辑你自己实现。函数签名都来自 `tests/adapters.py` 和 handout(公开接口)。
>
> **验证习惯**:每写完一个组件,接到 `tests/adapters.py` 对应的 `run_xxx`,然后跑它的测试。
> 通用命令:`uv run pytest <测试文件> -k "<关键字>" -v`

---

## 进度总览

图例:✅ 完成 / 🔨 进行中 / ⬜ 未开始 / ✍️ 书面题(无需写代码)

### 第 2 章 BPE Tokenizer
- [x] ✅ `train_bpe`(15分)— BPE 训练
- [x] ✅ `tokenizer`(15分)— encode/decode 类
- [ ] ✍️ `unicode1`(1分)、`unicode2`(3分)— Unicode 理解题
- [ ] ✍️ `train_bpe_tinystories`(2分)、`train_bpe_expts_owt`(2分)、`tokenizer_experiments`(4分)— 实验题

### 第 3 章 模型架构(核心代码)
- [x] ✅ `linear`(1分)
- [x] ✅ `embedding`(1分)
- [x] ✅ `rmsnorm`(1分)
- [x] ✅ `positionwise_feedforward` / SwiGLU(2分)
- [x] ✅ `softmax`(1分)
- [x] ✅ `rope`(2分)
- [x] ✅ `scaled_dot_product_attention`(5分)
- [x] ✅ `multihead_self_attention`(5分)
- [x] ✅ `transformer_block`(3分)
- [x] ✅ `transformer_lm`(3分)
- [ ] ✍️ `transformer_accounting`(5分)— 资源核算题

### 第 4-5 章 训练
- [ ] ⬜ `cross_entropy`(1分)
- [ ] ⬜ `adamw`(2分)
- [ ] ⬜ `learning_rate_schedule`(1分)
- [ ] ⬜ `gradient_clipping`(1分)
- [ ] ⬜ `data_loading`(2分)
- [ ] ⬜ `checkpointing`(1分)
- [ ] ⬜ `training_together`(4分)— 训练主循环
- [ ] ✍️ `learning_rate_tuning`(1分)、`adamw_accounting`(2分)

### 第 6-7 章 生成 & 实验
- [ ] ⬜ `decoding`(3分)— 文本生成
- [ ] ✍️ 实验类:`experiment_log`、TinyStories 调参、消融、OWT、leaderboard(需要 GPU)

---

## 推荐实现顺序(按依赖关系分层)

> 原则:**下层是上层的积木**,从下往上搭,每层内部可任意顺序。

### 第 1 层 · 独立小积木(无依赖)
写起来快,先把手感练熟。

| 顺序 | 组件 | 类/函数 | adapter | 测试关键字 |
|---|---|---|---|---|
| 1 | Linear ✅ | `class Linear(nn.Module)`:`__init__`, `forward` | `run_linear` | `linear` |
| 2 | Embedding ✅ | `class Embedding(nn.Module)`:`__init__`, `forward` | `run_embedding` | `embedding` |
| 3 | RMSNorm ✅ | `class RMSNorm(nn.Module)`:`__init__`, `forward` | `run_rmsnorm` | `rmsnorm` |
| 4 | SiLU + SwiGLU 🔨 | `class SwiGLU(nn.Module)`:`__init__`, `forward`(+ SiLU 函数) | `run_silu`, `run_swiglu` | `silu`, `swiglu` |
| 5 | Softmax 🔨 | 一个函数即可(无需类) | `run_softmax` | `softmax` |

### 第 2 层 · 注意力(依赖第 1 层)
本作业**最烧脑**的部分,慢慢来。

| 顺序 | 组件 | 类/函数 | adapter | 测试关键字 |
|---|---|---|---|---|
| 6 | RoPE 🔨 | `class RotaryPositionalEmbedding(nn.Module)`:`__init__`, `forward(x, token_positions)` | `run_rope` | `rope` |
| 7 | Scaled Dot-Product Attention | 函数/模块:输入 `Q,K,V,mask` | `run_scaled_dot_product_attention` | `scaled_dot_product` |
| 8 | Multi-Head Self-Attention | `class MultiHeadSelfAttention(nn.Module)`:`__init__`, `forward`(组合 7 + 因果mask + RoPE) | `run_multihead_self_attention`(+`_with_rope`) | `multihead` |

### 第 3 层 · 组装(依赖第 1、2 层)

| 顺序 | 组件 | 类/函数 | adapter | 测试关键字 |
|---|---|---|---|---|
| 9 | Transformer Block | `class TransformerBlock(nn.Module)`:`__init__`, `forward`(pre-norm:RMSNorm→注意力→残差→RMSNorm→FFN→残差) | `run_transformer_block` | `transformer_block` |
| 10 | Transformer LM | `class TransformerLM(nn.Module)`:`__init__`, `forward`(Embedding→N×Block→RMSNorm→输出Linear) | `run_transformer_lm` | `transformer_lm` |

### 第 4 层 · 训练组件(模型之外,大多是函数)

| 顺序 | 组件 | 类/函数 | adapter | 测试关键字 |
|---|---|---|---|---|
| 11 | Cross-Entropy | 函数 | `run_cross_entropy` | `cross_entropy` |
| 12 | AdamW | `class AdamW(torch.optim.Optimizer)`:`__init__`, `step` | `get_adamw_cls` | `adamw` |
| 13 | 余弦学习率调度 | 函数 | `run_get_lr_cosine_schedule` | `lr` / `schedule` |
| 14 | 梯度裁剪 | 函数 | `run_gradient_clipping` | `gradient_clipping` |
| 15 | 数据加载 | 函数 | `run_get_batch` | `get_batch` / `data` |
| 16 | Checkpoint | 两个函数:save / load | `run_save_checkpoint`, `run_load_checkpoint` | `checkpoint` |

### 第 5 层 · 整合 & 实验(依赖前面全部)
- 17. `training_together` — 把模型+优化器+数据+checkpoint 拼成训练主循环(脚本,非单元测试)
- 18. `decoding` — 用训好的模型生成文本(temperature / top-p 采样)
- 19. 实验题 — 跑训练、记录、写报告(需要 GPU 算力)

---

## 各组件的"要点提醒"(概念,非解法)

> 公式细节去 handout 对应的 Problem 小节看。这里只记**容易踩的坑**。

- **RMSNorm**:在最后一维求 `mean(x²)`,`eps` 放根号**里面**;先转 `float32` 算再转回原 dtype。
- **SwiGLU**:`W2·(SiLU(W1·x) ⊙ W3·x)`;`⊙` 是逐元素 `*`,不是矩阵乘;注意三个权重形状。
- **Softmax**:**先减最大值**再 exp(数值稳定);在指定 `dim` 上做,`keepdim=True`。
- **RoPE**:按 `(theta, d_k, max_seq_len)` 预计算 cos/sin;作用在 Q、K 上(不作用 V);按 `token_positions` 取对应角度。
- **Scaled Dot-Product Attention**:`softmax(QKᵀ/√d_k + mask)·V`;mask 在 softmax**之前**加,被遮位置设成极大负数。
- **Multi-Head**:Q/K/V 一次投影再拆成多头;**因果 mask**(下三角);可选 RoPE 作用在每个头的 Q、K 上。
- **Transformer Block**:**pre-norm** 结构(先归一化再进子层),两处残差连接。
- **Cross-Entropy**:为数值稳定,别先 softmax 再 log;用 log-sum-exp 技巧。
- **AdamW**:权重衰减是**解耦**的(直接作用在参数上,不进梯度的二阶矩)。

---

## Tokenizer 类接口备忘(已完成 ✅,留作参考)

`class Tokenizer`(handout `tokenizer` 题):
- `__init__(self, vocab, merges, special_tokens=None)`
- `from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)` — 类方法
- `encode(self, text) -> list[int]`
- `encode_iterable(self, iterable) -> Iterator[int]` — 省内存地处理大文件
- `decode(self, ids) -> str`

---

## 怎么用这份文档
1. 按"推荐实现顺序"从上往下做,做完在"进度总览"打勾。
2. 每个组件:写类/函数 → 接 `tests/adapters.py` 的 `run_xxx` → 跑测试关键字验证。
3. ✍️ 书面题和实验题可以穿插着做,别等到最后。
4. 卡住了,先回到 handout 对应 Problem 小节读公式,再问。
