import regex as re

class Tokenizer:
      
    # vocab is vocab = {i: bytes([i]} and add some BPE tokens
    # merges is list

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.bytes_to_ids = {b: i for i, b in vocab.items()}
    # texts -> ids    
    def encode(self, text: str):
        
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # 如果列表special_tokens不为空

        # tok = "<|endoftext|>"
        # re.escape(tok) → "\<\|endoftext\|\>"

        if self.special_tokens:
            sorted_tokens =sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(tok) for tok in sorted_tokens)
            chunks = re.split(f"({special_pattern})", text)
        else:
            chunks = [text]
        
        merged_ids = []

        for chunk in chunks:
            if chunk in self.special_tokens:
                token_byte = chunk.encode("utf-8")
                tok_id = self.bytes_to_ids[token_byte]
                merged_ids.append(tok_id)
                continue
   
            for match in re.finditer(PAT, chunk):
                piece = match.group()
                piece_bytes = piece.encode("utf-8")
                # 对每个词的bytes序列执行BPE合并
                parts = [bytes([b]) for b in piece_bytes]
                
                for a, b in self.merges:
                    i = 0
                    while i < len(parts) - 1:
                        if parts[i] == a and parts[i+1] == b:
                            merged = a + b
                            parts= parts[:i] + [merged] + parts[i+2:]
                        else:
                            i += 1
                # 将合并的bytes转化成ids
                piece_ids = [self.bytes_to_ids[part] for part in parts]
                merged_ids.extend(piece_ids)
    
        return merged_ids
    
    # ids -> texts

    def decode(self, ids: list):
        bytes_list = [self.vocab[i] for i in ids if i in self.vocab]
        return b"".join(bytes_list).decode("utf-8", errors="replace")
    
    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)