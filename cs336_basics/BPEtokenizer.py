import regex as re



def train_bpe(input_path, vocab_size, special_tokens):
    # 按照这个来划分token
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # 打开文件
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    split_token = "|".join(re.escape(token) for token in special_tokens)
    parts = re.split(split_token, text)



    # 这里面放着所有的token的张量
    all_tokens = []

    # encode 原始的词表
    # 遍历每一段话（根据"<|endoftext|>"分割开）
    for part in parts:
        if part == "":
            continue
        # 根据这个分词规则分每一个token
        for match in re.finditer(PAT, part):
            # 将每一个token转换成byte
            # 将这个token的每一个byte存在一个数组中
            # 比如"hello" → [104, 101, 108, 108, 111]
            token_list = list(match.group().encode('utf-8'))
            all_tokens.append(token_list)

    # 建立词汇表和建立词汇表大小
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')

    merges = []
    while len(vocab) < vocab_size:
        pair_counts = {}
        # 单独处理每一个token

        for token_list in all_tokens:
            for i in range(len(token_list) - 1):
                first_byte = token_list[i]
                second_byte = token_list[i+1]
                cur_pair = (first_byte, second_byte)
                pair_counts[cur_pair] = pair_counts.get(cur_pair, 0) + 1
        
        # 所有的token统计完了之后寻找一个出现次数最多的
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]]))[0]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        new_idx = len(vocab)

        # 用这个新的替换旧的
        for i in range(len(all_tokens)):
            old_list = all_tokens[i]
            # new_list is a new list to replece the old_list
            new_list = []
            j = 0
            while j < len(old_list):
                if j+1 < len(old_list) and old_list[j] == best_pair[0] and old_list[j+1] == best_pair[1]:
                    new_list.append(new_idx)
                    j += 2
                else:
                    new_list.append(old_list[j])
                    j += 1
            all_tokens[i] = new_list
        vocab[new_idx] = vocab[best_pair[0]] + vocab[best_pair[1]]
    return vocab, merges
    

    
    
