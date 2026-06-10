import numpy as np
import pickle
from cs336_basics.tokenizer import Tokenizer


    
def process_file(txt_path, bin_path, tokenizer, batch_size=1_000_000):
    print(f"Processing {txt_path} -> {bin_path}")
    buffer = []
    total = 0

    open(bin_path, "wb").close()
    
    with open(txt_path, "r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            buffer.append(token_id)

            if len(buffer) >= batch_size:
                arr = np.array(buffer, dtype=np.uint16)
                with open(bin_path, "ab") as f_out:
                    arr.tofile(f_out)
                total += len(buffer)
                buffer = []
                print(f"Written {total} tokens...")
    if buffer:
        arr = np.array(buffer, dtype=np.uint16)
        with open(bin_path, "ab") as f_out:
            arr.tofile(f_out)
        total += len(buffer)

    print(f"Done! Total tokens: {total}\n")
        
def main():
    with open("data/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("data/merges.pkl", "rb") as f:
        merges = pickle.load(f)
    
    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=["<|endoftext|>"]
    )
    process_file("data/TinyStoriesV2-GPT4-train.txt", "data/train.bin", tokenizer)

if __name__ == "__main__":
    main()