import torch

def load_dataset():
    with open('/home/coby/Repositories/mlfundamentals/karpathy/flaubertgpt/bovary.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    with open('/home/coby/Repositories/mlfundamentals/karpathy/flaubertgpt/salammbo.txt', 'r', encoding='utf-8') as f:
        text += f.read()
    return text

def train_test_split(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = { ch : i for i, ch in enumerate(chars) }
    itos = { i : ch for i, ch in enumerate(chars) }

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(text))
    train_data = data[:n]
    test_data = data[n:]
    return train_data, test_data

def get_vocab_size(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    return vocab_size