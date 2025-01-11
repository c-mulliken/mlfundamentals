import torch
import torch.nn as nn
from torch.nn import functional as F
import sys


def get_batch(split, train_data, test_data, batch_size, block_size):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, test_data, batch_size, block_size):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, test_data, batch_size, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(model, train_data, test_data, max_iters, lr, batch_size, block_size,
          eval_interval, eval_iters):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, eval_iters, train_data, test_data, batch_size, block_size)
            print(f"step: {step}, train loss : {losses['train']:.4f}, test loss : {losses['val']:.4f}")

        xb, yb = get_batch('train', train_data, test_data, batch_size, block_size)

        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def generate(model, length, device, decode):
    return decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=length)[0].tolist())