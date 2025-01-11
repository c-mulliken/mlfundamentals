def load_dataset():
    with open('karpathy/flaubertgpt/bovary.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    return text