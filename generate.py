import torch
import pickle

from cs336_basics.TransformerLM import TransformerLM 
from cs336_basics.tokenizer import Tokenizer

def setup():
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ----------------------
    # 【填空1】加载 vocab.pkl、merges.pkl，初始化Tokenizer
    # ----------------------
    # 读取词表与合并规则
    with open("data/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("data/merges.pkl", "rb") as f:
        merges = pickle.load(f)
    
    # 实例化Tokenizer，传入特殊token
    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=["<|endoftext|>"]
    )

        
    vocab_size = 1000
    context_length = 256
    d_model = 512
    num_layers = 4
    num_heads = 8
    d_ff = 1344
    rope_theta = 10000.0
    eps = 1e-5

    # 初始化Transformer模型
    model = TransformerLM(
        vocab_size = vocab_size,
        context_length = context_length,
        d_model = d_model,
        num_layers = num_layers,
        num_heads = num_heads,
        d_ff = d_ff,
        rope_theta = rope_theta
    )

    def load_checkpoint(model, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    
    load_checkpoint(model, ckpt_path="./checkpoints/latest.pt")

    model.eval()
    model.to(device)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "context_length": context_length,
    }

# ======================
# 第二块：generate 核心生成函数
# ======================
@torch.no_grad()
def generate(
    model,
    token_ids,
    tokenizer,
    device,
    context_length,
    max_new_tokens=100,
    temperature=1.0,
    top_p=0.9,
    use_greedy=True
):
    # 获取结束符token id
    eos_id = tokenizer.bytes_to_ids["<|endoftext|>".encode("utf-8")]

    # ----------------------
    # 【填空4】token id列表转张量，增加batch维度，迁移到device
    # 目标形状: (1, 序列长度)
    # ----------------------
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    input_ids = input_ids.unsqueeze(0)  # 增加batch维
    input_ids = input_ids.to(device)  # 迁移设备

    # 循环生成新token
    for _ in range(max_new_tokens):
        # ----------------------
        # 【填空5】序列截断：长度超出上下文上限则保留最后context_length个token
        # ----------------------
        if input_ids.shape[1] > context_length:
            input_ids = input_ids[:, -context_length:]

        # ----------------------
        # 【填空6】模型前向传播，得到logits
        # ----------------------
        logits = model(input_ids)

        # ----------------------
        # 【填空7】只取最后一个位置的logits，形状 (1, vocab_size)
        # ----------------------
        next_logits = logits[:, -1, :]

        # ----------------------
        # 【填空8】根据模式选择下一个token
        # ----------------------
        if use_greedy:
            # 贪心解码：取概率最大的id
            next_token = next_logits.argmax(dim=-1)
        else:
            # 1. 温度缩放
            next_logits = next_logits / temperature
            # 2. 转为概率分布
            probs = torch.softmax(next_logits, dim=-1)

            # 3. Top-P 核采样
            # 概率降序排序，保留原索引
            probs_sort, indices_sort = torch.sort(probs, descending=True)
            # 计算累积概率
            cum_probs = torch.cumsum(probs_sort, dim=-1)
            # 掩码：超出top-p的位置置0
            mask = cum_probs - probs_sort > top_p
            probs_sort[mask] = 0.0
            # 概率重新归一化
            probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True)
            # 采样
            next_idx = torch.multinomial(probs_sort, num_samples=1)
            # 映射回原始token id
            next_token = torch.gather(indices_sort, -1, next_idx)

        # ----------------------
        # 【填空9】将新token拼接到原序列末尾
        # ----------------------
        next_token = next_token.unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # 遇到结束符，提前终止生成
        if next_token.item() == eos_id:
            break

    # 转为普通列表返回
    return input_ids[0].tolist()

# ======================
# 第三块：主程序 调用入口
# ======================
if __name__ == "__main__":
    # 初始化环境与模型
    components = setup()
    model = components["model"]
    tokenizer = components["tokenizer"]
    device = components["device"]
    context_length = components["context_length"]

    # ----------------------
    # 【填空10】定义提示词prompt
    # ----------------------
    prompt = "Once upon a time"


    # ----------------------
    # 【填空11】prompt编码为token id列表
    # ----------------------
    prompt_ids = tokenizer.encode(prompt)

    # 调用生成函数
    generated_ids = generate(
        model=model,
        token_ids=prompt_ids,
        tokenizer=tokenizer,
        device=device,
        context_length=context_length,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        use_greedy=True  # 先开贪心，跑通后再关闭
    )

    # ----------------------
    # 【填空12】id列表解码为文本并打印
    # ----------------------
    output_text = tokenizer.decode(generated_ids)
    print("\n=== 生成结果 ===")
    print(output_text)