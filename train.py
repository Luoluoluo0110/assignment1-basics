import argparse
import numpy as np
import torch
import os

# ======================
# 只从核心包导入
# ======================
from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.Adamw import Adamw
from cs336_basics.utils import (
    gradient_clipping,
    lr_cosine_schedule,
    cross_entropy
)
from cs336_basics.data import get_batch
from cs336_basics.serialization import save_checkpoint, load_checkpoint

def main():
    # ======================
    # 1. 解析命令行参数
    # ======================
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--d_model", type=int, default=512, help="Model hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344, help="FFN intermediate dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--total_steps", type=int, default=2500, help="Total training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="LR warmup steps")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max L2 norm for gradient clipping")

    # Paths
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--train_bin_path", type=str, default="./data/train.bin")
    parser.add_argument("--val_bin_path", type=str, default="./data/val.bin")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ======================
    # 2. Load data
    # ======================
    train_tokens = np.memmap(args.train_bin_path, dtype=np.uint16, mode="r")
    val_tokens = np.memmap(args.val_bin_path, dtype=np.uint16, mode="r")

    # ======================
    # 3. Model
    # ======================
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    # ======================
    # 4. Optimizer
    # ======================
    optimizer = Adamw(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # ======================
    # 5. Resume checkpoint
    # ======================
    start_step = 0
    ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
    if os.path.exists(ckpt_path):
        print("Loading checkpoint...")
        start_step = load_checkpoint(ckpt_path, model, optimizer)

    # ======================
    # 6. Training loop
    # ======================
    model.train()
    val_loss = None  # 初始化，防止 resume 未定义

    for step in range(start_step, args.total_steps):
        # ======================
        # 学习率调度（正确顺序）
        # ======================
        lr = lr_cosine_schedule(
            it=step,
            max_learning_rate=args.learning_rate,
            min_learning_rate=args.learning_rate * 0.1,
            warmup_iters=args.lr_warmup_steps,
            cosine_cycle_iters=args.total_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch
        x, y = get_batch(train_tokens, args.batch_size, args.context_length, device)

        # Forward
        logits = model(x)

        # Loss  view → reshape
        B, S, V = logits.shape
        logits_flat = logits.reshape(B * S, V)
        y_flat = y.reshape(B * S)
        loss = cross_entropy(logits_flat, y_flat)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        #  梯度裁剪使用命令行参数
        gradient_clipping(model.parameters(), max_l2_norm=args.max_grad_norm)

        optimizer.step()

        # ======================
        # 验证集评估
        # ======================
        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(val_tokens, args.batch_size, args.context_length, device)
                logits_val = model(x_val)
                val_loss = cross_entropy(logits_val.reshape(-1, args.vocab_size), y_val.reshape(-1))
            model.train()

        # ======================
        #  安全打印：val_loss 存在才打印
        # ======================
        if step % 10 == 0:
            if val_loss is not None:
                print(f"Step {step:4d} | Train Loss {loss.item():.4f} | Val Loss {val_loss.item():.4f} | LR {lr:.6f}")
            else:
                print(f"Step {step:4d} | Train Loss {loss.item():.4f} | Val Loss ---- | LR {lr:.6f}")

        # Save checkpoint
        if step % 100 == 0:
            save_checkpoint(model, optimizer, step + 1, ckpt_path)


if __name__ == "__main__":
    main()