import torch
import torch.nn as nn
from torch.nn import functional as F

"""
HeartGPT Fine-tuning Model

Based on HeartGPT by Harry Davies (09/2024):
https://github.com/harryjdavies/HeartGPT/blob/main/HeartGPT_finetuning.py

Which builds on Andrej Karpathy's transformer implementation:
https://github.com/karpathy/ng-video-lecture

Paper: Davies et al. (2024) "Interpretable Pre-Trained Transformers for Heart Time-Series Data" 
https://arxiv.org/abs/2407.20775
"""


class Head(nn.Module):
    def __init__(self, cfg, head_size, mask=True):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.mask = mask
        self.register_buffer(
            "tril", torch.tril(torch.ones((cfg.block_size, cfg.block_size)))
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        if self.mask:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, head_size, mask=True):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(cfg, head_size, mask=mask) for _ in range(cfg.n_head)]
        )
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg, mask=True):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.sa = MultiHeadAttention(cfg, head_size, mask=mask)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class ClassificationHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(cfg.n_embd, 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.classifier(x)
        x = x[:, -1, :]  # Take last token for classification
        x = x.squeeze(-1)  # Remove last dimension
        return x


class Heart_GPT_FineTune(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        # mask option in blocks allows you to unmask the last layer if set to False
        self.blocks = nn.Sequential(
            *[Block(cfg) for _ in range(cfg.n_layer - 1)] + [Block(cfg)]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.cfg.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
