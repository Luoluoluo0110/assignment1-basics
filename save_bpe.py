# 训练 BPE,存 vocab/merges 
# save_bpe.py
import pickle
from cs336_basics.BPEtokenizer import train_bpe

if __name__ == "__main__":
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"]
    )

    with open("data/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("data/merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    print("Saved vocab.pkl and merges.pkl!")